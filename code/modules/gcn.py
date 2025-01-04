from torch_geometric.nn import GCNConv, SimpleConv, GATConv
from torch.functional import F
from transformers import AutoModel
from torch import nn
import torch

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

class GCN(torch.nn.Module):
    def __init__(self, in_feats=None, n_classes=15, n_hidden=200):
        super().__init__()
        torch.manual_seed(1234567)
        self.in_feats = in_feats
        self.conv1 = GCNConv(in_feats, 60)
        self.conv2 = GCNConv(60, n_classes)

        # Note that SimpleConvs are non-trainable!
        # They are simple, but efficient, checkout here for literature / more reference implementations:
        # https://github.com/Tiiiger/SGC?tab=readme-ov-file
        # self.conv1 = SimpleConv(aggr='sum', 
        #                         # combine_root="self_loop" # we don't need self loop, since we include bert encoded embedding.
                            # ) # See: https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.SimpleConv.html#torch_geometric.nn.conv.SimpleConv
        # self.conv2 = SimpleConv(aggr='sum', 
        #                         # combine_root="self_loop"
                                # )
        self.linear = torch.nn.Linear(self.in_feats, n_classes)
    def forward(self, embeds, edge_index):
        # edge-index is LIL-encoded adjacency matrix
        # import pdb; pdb.set_trace()
        x = self.conv1(embeds, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        x = self.conv2(x, edge_index)
        # import pdb; pdb.set_trace()
        # x = self.linear(x)
        return x

class BertDialGCN(torch.nn.Module):
    def __init__(self, pretrained_model, no_classes=15, m=0.7, n_hidden=200, dropout=0.5, device="cuda:0"):
        super(BertDialGCN, self).__init__()
        self.m = m
        self.no_classes = no_classes
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # self.deberta_model = pretrained_model
        self.deberta_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.deberta_model.modules())[-2].embedding_dim
        self.classifier = torch.nn.Linear(self.feat_dim, no_classes)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_classes=no_classes,
            n_hidden=n_hidden,
            # n_classes=no_classes,
            # n_layers=gcn_layers-1,
            # activation=F.elu,
            # dropout=dropout
        )
        self.device = device
        # self.gcn = GAT(
        #     num_layers=1,
        #     in_dim=self.feat_dim,
        #     num_hidden=n_hidden,
        #     num_classes=no_classes,
        #     # heads=[8, 1],
        #     heads=[8] * (2-1) + [1],
        #     activation=F.elu
        #     # feat_drop=dropout,
        #     # attn_drop=dropout,
        #     # negative_slope=0.2,
        #     # residual=False
        # )

    def forward(self, g, idx):
        input_ids, attention_mask = g.x[idx][0].unsqueeze(dim=0), g.x[idx][1].unsqueeze(dim=0)
        # import pdb; pdb.set_trace()
        # feats = self.deberta_model(input_ids, attention_mask)[0][:, 0]
        if self.training:
            # import pdb; pdb.set_trace()
            # print("Input ID device", input_ids.get_device())
            # print("Attention Mask device", attention_mask.get_device())
            # print("Graph", )
            feats = self.deberta_model(input_ids.to(self.device), 
                                       attention_mask.to(self.device)
                                    ).last_hidden_state[:,0][0]
            # import pdb; pdb.set_trace()
            # g.node_embeddings.detach_()
            # g.node_embeddings[idx] = feats # store the updated embeddings for the GraphNN
        else:
            feats = self.deberta_model(input_ids.to(self.device), 
                                       attention_mask.to(self.device)
                                    ).last_hidden_state[:,0][0]
            # feats = g.node_embeddings[idx] # fetch if in eval mode
        cls_logit = self.classifier(feats)
        # import pdb; pdb.set_trace()
        cls_pred = torch.nn.Softmax(dim=0)(cls_logit) # TODO: Confirm that this is the right dimension
        # gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        # gcn_logit = self.gcn(g.node_embeddings, g.edge_index, idx) # Obtain logis from Siple Graph Conv
        gcn_logit = self.gcn(g.node_embeddings, g.edge_index.to(self.device))[idx] # Obtain logis from GAT
        gcn_pred = torch.nn.Softmax(dim=0)(gcn_logit) # TODO: Confirm that this is the right dimension
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred