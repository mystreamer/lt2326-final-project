import os
import json
from tqdm import tqdm
import copy
import torch
import numpy as np
from torch_geometric.data import Data
from functools import partial
import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)


def get_node_embedding(model, node, device):
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        return model(node[0].unsqueeze(dim=0).to(device), 
                             attention_mask=node[1].unsqueeze(dim=0).to(device), 
                             output_hidden_states=True
                        ).last_hidden_state[:,0][0].clone().detach()

def prefill_node_embeddings(model, nodeset_graphs, device):
    for data in tqdm(nodeset_graphs):
        data.node_embeddings = torch.vstack([get_node_embedding(model, node, device) for node in data.x])
    return nodeset_graphs

def get_partial_tok(tokenizer, model_max_length):
    return partial(tokenizer, 
                is_split_into_words=False, 
                truncation=True, 
                padding="max_length", 
                max_length=model_max_length, 
                return_tensors="pt"
            )

def encode_node_text(partial_tok,  node):
    """Encodes a string of text into a tokenised form

    Args:
        node (Dict): A dict of type node in AIF (Argument Interchange Format)

    Raises:
        ValueError: Raised the node type attribute is not recognised

    Returns:
        node (Dict): A node enriched with tokeniser-encoded "tokens" field (PyTorch tensor)
    """
    n = copy.deepcopy(node)
    if n["type"] in ["I", "L"]:
        n["tokens"] = partial_tok(text=n["text"])
    else:
        if n["type"] in ["CA", "RA", "MA"]: # An S-node
            n["tokens"] = partial_tok(text="[S]")
            n["label"] = n["text"]
        elif n["type"] in ["YA"]: # An Illocutionary-node
            n["tokens"] = partial_tok(text="[YA]") 
            n["label"] = n["text"]
        elif n["type"] in ["TA"]: # A transitional node
            n["tokens"] = partial_tok(text="[TA]")
            # n["label"] = n["text"] # No label needed for transitional nodes
        else:
            raise ValueError("Unknown node type")
    return n

# Load json file and pprint
def load_json_as_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_all_nodesets(nodeset_path):
    nodesets = []
    for filename in os.listdir(nodeset_path):
        if filename.endswith(".json"):
            nodeset = load_json_as_dict(f"{nodeset_path}/{filename}")
            nodeset["filename"] = filename
            nodesets.append(nodeset)
    return nodesets

# Build a feature matrix in the form [num_nodes, num_node_features (tokeniser encoded strings?)]
def nodeset_to_pyg_data(nodeset):
    nodes = nodeset["nodes"]
    # edges = nodeset["edges"]
    # locutions = nodeset["locutions"]
    nodeid_to_index = {n["nodeID"]: i for i, n in enumerate(nodes)}
    nodeid_to_labels = {n["nodeID"]: n["label"] for n in nodes if "label" in n}
    node_features = torch.concat([torch.vstack(
        [n["tokens"][k] for k in ["input_ids", "attention_mask"]]
        ).unsqueeze(dim=0) for n in nodes])
    return node_features, nodeid_to_index, nodeid_to_labels

# Encode the edges (directed) in the COO format (I guess the adjacency matrix)
def construct_edge_index(nodeset, nodeid_to_index, symmetrical=False):
    """Construct an edge index in COO format from a nodeset
        Note: We use tuple representation and convert it to COO format on return, as detailed here:
        https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs

    Args:
        nodeset (Dict): A nodeset
    
    Returns:
        edge_index (Tensor): A 2xN tensor of edge indices in COO format
    """
    edges = nodeset["edges"]
    def generate_edges_in_tuple_format(edges, symmetrical=False):
        for e in edges:
            yield ([nodeid_to_index[e["fromID"]], nodeid_to_index[e["toID"]]], [nodeid_to_index[e["toID"]], nodeid_to_index[e["fromID"]]]) \
                if symmetrical else ([nodeid_to_index[e["fromID"]], nodeid_to_index[e["toID"]]], )
    return torch.tensor([y for x in generate_edges_in_tuple_format(edges, symmetrical) for y in x], dtype=torch.long)

def generate_pyg_dataset(encoded_nodesets):
    """Generates a PyG dataset from a list of encoded nodesets

    Args:
        encoded_nodesets (List[Dict]): A list of encoded nodesets

    Returns:
        dataset (List[Data]): A list of PyG Data objects
    """
    dataset = []
    for nodeset in encoded_nodesets:
        x, nodeid_to_index, nodeid_to_labels = nodeset_to_pyg_data(nodeset)
        edge_index = construct_edge_index(nodeset, nodeid_to_index, symmetrical=False)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        data.nodeid_to_index = nodeid_to_index
        data.index_to_nodeid = {v: k for k, v in nodeid_to_index.items()}
        data.nodeid_to_labels = nodeid_to_labels
        dataset.append(data)
    return dataset

def generate_individual_nodes_dataset(nodeset_graphs):
    """
        Generates a dataset of individual nodes, i denotes the graph number
        and j denotes the node number in the graph.
        The last element in the tuple is the label of the node.
    """
    ds = []
    for i, graph in enumerate(nodeset_graphs):
        for j, unlabelled_node in enumerate(graph.nodeid_to_labels.items()):
            # Perform classification only on three nodes
            # import pdb; pdb.set_trace()
            # if unlabelled_node[1] in ["Default Inference", "NONE"]: # reduce to two labels
            dp = (i, graph.nodeid_to_index[unlabelled_node[0]],unlabelled_node[1])
            ds.append(dp)
    return ds