import os
import torch
import copy
import json
import pickle
import argparse
from itertools import cycle
from torch.functional import F
from modules.gcn import BertDialGCN
from external.prepare_data import main
from modules.utils import detect_platform 
from transformers import DebertaV2TokenizerFast
from external.eval_official import main as eval_off

from modules.data_processing import (
    load_all_nodesets,
    generate_pyg_dataset,
    generate_individual_nodes_dataset,
    get_partial_tok,
    encode_node_text,
    prefill_node_embeddings
)

# Test Loop
def evaluate(model, test_loader, ohe, num_labels):
    unencoded_predictions = []
    model.eval()
    loss_r = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            # print("iteration")
            nodes, labels = tuple(t for t in batch)
            print("nodes", nodes)
            print("labels", labels)
            outputs = model(nodeset_graphs[nodes[0]], nodes[1])
            # loss = criterion(outputs, labels)
            # loss_r += loss.item()
            # import pdb; pdb.set_trace()
            print("Outputs", outputs)
            _, predicted = torch.max(outputs.data.unsqueeze(dim=0), 1)
            # total += outputs.size(0)
            predicted = predicted.cpu()
            # total += 1
            # print("Size", outputs.size(0))
            # _, labels_maxed = torch.max(labels.data.unsqueeze(dim=0), 1)
            # print("Predicted: ", predicted, "Labels: ", labels_maxed)
            # correct += (predicted == labels_maxed).sum().item()
            print(predicted)
            unencoded_predictions.append(ohe.inverse_transform(F.one_hot(torch.LongTensor([predicted]), num_labels))[0][0])

    # avg_loss = loss_r / len(test_loader)
    # accuracy = 100 * correct / total
    # print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return unencoded_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    if not args.debug:
        # Evaluate model on the gold-standard
        os.makedirs("./data/normalised_eval", exist_ok=True)
        model_name = "microsoft/deberta-v3-small"
        model_max_len = 384

        device = detect_platform(0)
    
        nt_tokens = ["[S]", "[YA]", "[TA]"]
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
        tokenizer.add_tokens(nt_tokens) # add new special tokens
        partial_tok = get_partial_tok(tokenizer, model_max_len)

        # Pre-normalise the embeddings
        main(
            input_dir="./data/qt-30/eval",
            output_dir="./data/normalised_eval",
            integrate_gold_data=False, # False in the case of evaluation
            nodeset_id=None,
            nodeset_blacklist=None,
            nodeset_whitelist=None,
            s_node_type="RA",
            s_node_text="NONE",
            ya_node_text="NONE",
        )

        # Run inference on the nodeset
        nodeset_path = "./data/normalised_eval"
        nodesets = load_all_nodesets(nodeset_path)

        with open(f'./models/bert-gcn.ohe', "rb") as f:
            data = pickle.load(f)
            ohe = data["encoder"]
            num_labels = ohe.categories_[0].shape[0] 

        # Load the model
        model = BertDialGCN(no_classes=num_labels, m=0.7, pretrained_model=model_name, device=device).to(device)
        model.load_state_dict(torch.load("./models/bert-gcn.pth"))
        model.eval()

        encoded_nodesets = list(map(lambda nodeset: {
            "nodes": list(map(encode_node_text, cycle([partial_tok]), nodeset["nodes"])),
            "edges": nodeset["edges"],
            "locutions": nodeset["locutions"]
        }, nodesets))

        nodeset_graphs = generate_pyg_dataset(encoded_nodesets)
        nodeset_graphs = prefill_node_embeddings(model.deberta_model, nodeset_graphs, device)
        # Prepare data for model
        X = generate_individual_nodes_dataset(nodeset_graphs)
        loader = list(zip(X, range(len(X))))


        res = evaluate(model, loader, ohe, num_labels) 
        X_pred = list(zip([(x[0], x[1]) for x in X], res))
        # only include relevant nodesets, i.e. those with predicted labels
        relevant_nodeset_ids = sorted(list(set([x[0][0] for x in X_pred])))
        nodesets_pred = copy.deepcopy(nodesets)
        # import pdb; pdb.set_trace()
        # replace the nodes with the predicted labels
        def node_swap(ns_pred):
            """
            Replace the gold labels with the predicted labels in the nodesets
            """
            for node in ns_pred:
                print("Replacing text of node", node[0][1], "in nodeset", node[0][0])
                nodesets_pred[node[0][0]]["nodes"][node[0][1]]["text"] = node[1]

        node_swap(X_pred)

        # subset only relevant nodesets
        nodesets_pred_rel = [nodesets_pred[id] for id in relevant_nodeset_ids]

        # Save the predicted nodesets to file:
        os.makedirs("./data/eval_predictions", exist_ok=True)
        for ns in nodesets_pred_rel:
            with open(f"./data/eval_predictions/{ns['filename']}", 'w') as f:
                json.dump(ns, f)

    # Pre-normalise the embeddings
    os.makedirs("./data/denormalised_eval", exist_ok=True)
    main(
        input_dir="./data/eval_predictions",
        output_dir="./data/denormalised_eval",
        integrate_gold_data=True, # False in the case of evaluation
        nodeset_id=None,
        nodeset_blacklist=None,
        nodeset_whitelist=None,
        s_node_type="RA",
        s_node_text="NONE",
        ya_node_text="NONE",
        re_revert_ra_relations=True,
        re_remove_none_relations=True,
    ) 
    # eval arguments
    eval_off(
        predictions_dir="./data/denormalised_eval",
        gold_dir="./data/qt-30/eval",
        mode="arguments"
    )
    # eval arguments
    eval_off(
        predictions_dir="./data/denormalised_eval",
        gold_dir="./data/qt-30/eval",
        mode="illocutions"
    )