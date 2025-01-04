import copy
import torch
import pickle
import argparse
from tqdm import tqdm 
from itertools import cycle
import torch.nn as nn
import torch.optim as optim
from functools import partial
from modules.utils import detect_platform
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModel, DebertaV2TokenizerFast


# Local imports
from modules.gcn import BertDialGCN
from modules.data_processing import (
    load_all_nodesets,
    generate_pyg_dataset,
    generate_individual_nodes_dataset,
    get_partial_tok,
    encode_node_text,
    prefill_node_embeddings,
)

def one_hot_encode_labels(x: list):
    return torch.tensor(ohe.transform([[node[2]] for node in x]).toarray(), dtype=torch.float32)

def train(
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        n_epoch=10, 
        iter_print=100,
        device="cpu"):
    
    # prev_loss = np.inf
    for epoch in tqdm(range(n_epoch)):
        model.train()
        loss_r = 0.0
        count_r = 0
        validation_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader)):
            # print(images.shape)
            # import pdb; pdb.set_trace()
            nodes, labels = tuple(t for t in batch)
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            print("Training", nodes, labels)
            # nodeset_graphs[nodes[0]].x[0][0].to(DEVICE)
            # nodeset_graphs[nodes[0]].x[0][1].to(DEVICE)
            outputs = model(nodeset_graphs[nodes[0]], nodes[1])
            # print(outputs)
            loss = criterion(outputs, labels.to(device))
            # import pdb; pdb.set_trace()
            loss.backward()
            # print("works once!")
            print(loss)
            optimizer.step()
            nodeset_graphs[nodes[0]].node_embeddings.detach_() # TODO: Is this purposeful?
            loss_r += loss.item()
            count_r += 1
            if (i+1) % iter_print == 0:
                print(f"Epoch [{epoch+1}/{n_epoch}], Step [{i+1}/{len(train_loader)}], Average Loss: {loss_r/count_r:.4f}")
                loss_r = 0.0
                count_r = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                nodes, labels = tuple(t for t in batch)
                # print("Validation", nodes, labels)
                outputs = model(nodeset_graphs[nodes[0]], nodes[1])
                loss = criterion(outputs, labels.to(device))
                validation_loss += loss.item()
        # Print loss after each epoch
        epoch_loss = loss_r / len(train_loader)
        print(f"\nEnd of Epoch {epoch+1}/{n_epoch}, Average Epoch Train Loss: {epoch_loss:.4f}")
        print(f"Average Validation Loss: {validation_loss / len(X_val):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    # Get the normalised nodesets
    device = detect_platform(0)# assume to run one cuda:0
    nodeset_path = "./data/normalised"
    nodesets = load_all_nodesets(nodeset_path)

    model_name = "microsoft/deberta-v3-small"
    model_max_len = 384

    nt_tokens = ["[S]", "[YA]", "[TA]"]

    # Get the pretrained instances
    # Run the tokeniser
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
    tokenizer.add_tokens(nt_tokens) # add new special tokens
    partial_tok = get_partial_tok(tokenizer, model_max_len)
    
    # Load the model
    deberta_model = AutoModel.from_pretrained(model_name).to(device)

    encoded_nodesets = list(map(lambda nodeset: {
        "nodes": list(map(encode_node_text, cycle([partial_tok]), nodeset["nodes"])),
        "edges": nodeset["edges"],
        "locutions": nodeset["locutions"]
    }, nodesets))

    # Make the dataset appropriately graph-structured
    nodeset_graphs = generate_pyg_dataset(encoded_nodesets)

    if args.debug:
        nodeset_graphs = nodeset_graphs[:10]

    # Get corresponding vector embeddings
    deberta_model.eval()

    nodeset_graphs = prefill_node_embeddings(deberta_model, nodeset_graphs, device)

    # Prepare data for model
    X = generate_individual_nodes_dataset(nodeset_graphs)

    # Encode the labels
    possible_labels = [[l] for l in sorted(list(set([node[2] for node in X])))]
    ohe = OneHotEncoder().fit(possible_labels)
    y = one_hot_encode_labels(X)
    num_labels = ohe.categories_[0].shape[0]
    # LABEL_NUMBER
    # Save the one-hot encoder
    with open(f'./models/bert-gcn.ohe', "wb") as f: 
        pickle.dump({ "encoder": ohe,
                     }, f)

    # Create train-test-split
    # Basic idea: Create the train-test-split by the indices of the nodesets
    ns_idx = list(range(len(nodeset_graphs)))
    X_train_idx, X_test_idx = train_test_split(ns_idx, test_size=0.2, random_state=1)
    X_train_idx, X_val_idx = train_test_split(X_train_idx, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    
    # Allocate nodes given the graph indices:
    # train:
    X_train = [node for node in X if node[0] in X_train_idx]
    y_train = torch.index_select(y, 0, torch.LongTensor([i for i, node in enumerate(X) if node[0] in X_train_idx]))
    # test dataset
    X_test = [node for node in X if node[0] in X_test_idx]
    y_test = torch.index_select(y, 0, torch.LongTensor([i for i, node in enumerate(X) if node[0] in X_test_idx]))
    # validation dataset
    X_val = [node for node in X if node[0] in X_val_idx]
    y_val = torch.index_select(y, 0, torch.LongTensor([i for i, node in enumerate(X) if node[0] in X_val_idx]))
    
    train_loader = list(zip(X_train, y_train))
    val_loader = list(zip(X_val, y_val))
    test_loader = list(zip(X_test, y_test))

    del deberta_model

    # Initialise GCN model
    model = BertDialGCN(no_classes=num_labels, m=0.7, pretrained_model=model_name, device=device).to(device)

    # Set optimizer, criterion

    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9

    # training loop
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    train(
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        n_epoch=5, 
        iter_print=100,
        device=device
        )
    print("Training complete!")

    torch.save(model.state_dict(), "./models/bert-gcn.pth")
    print("Model saved to disk!")