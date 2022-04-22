import argparse
import torch
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset
from attack_utils import multi_test_attack, select_nodes, single_test, get_classification_margin
from deeprobust.graph.targeted_attack import Nettack
from os import path
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--n_ptb', type=int, default=None,
                    help='Number of perturbations. If not set, will default to node degree + 2.')
parser.add_argument('--n_hidden', type=int, default=16, help='Dimension of node embeddings.')
parser.add_argument('--n_iters', type=int, default=200, help='Number of training iterations.')
parser.add_argument('--poisoning', action='store_true', help='If set, will perform poisoning attack.')
parser.add_argument('--results_dir', type=str, default='.', help='Path of directory where results should be saved.')
parser.add_argument('--no_attack', action='store_true', help='If set, no attack will be run. Attack arguments will '
                                                             'be ignored.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.no_attack:
    save_path = path.join(args.results_dir,
                          "no_attack",
                          args.dataset,
                          str(args.seed))
else:
    save_path = path.join(args.results_dir,
                          "poisoning" if args.poisoning else "evasion",
                          args.dataset,
                          str(args.seed))
if not path.exists(save_path):
    os.makedirs(save_path)
print("Results will be saved at: ", save_path)


def main():
    data = Dataset(root='/tmp/', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    surrogate = GCN(nfeat=features.shape[1],
                    nhid=args.n_hidden,
                    nclass=labels.max().item() + 1,
                    dropout=0,
                    lr=0.01,
                    weight_decay=0,
                    with_relu=False,
                    with_bias=False,
                    device=device)

    target_gcn = GCN(nfeat=features.shape[1],
                     nhid=args.n_hidden,
                     nclass=labels.max().item() + 1,
                     dropout=0,
                     lr=0.01,
                     weight_decay=0,
                     with_relu=True,
                     with_bias=False,
                     device=device)

    if args.no_attack:
        print("No attack!")
        target_gcn.initialize()
        target_gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=args.n_iters, patience=30, verbose=True)
        cnt = 0
        cls_margins = []
        node_list = select_nodes(surrogate, data)
        for target_node in tqdm(node_list):
            acc = single_test(adj, features, labels, target_node,
                              gcn=target_gcn, poisoning=False,
                              idx_train=idx_train, idx_val=idx_val,
                              train_iters=args.n_iters)
            if acc == 0:
                cnt += 1
            margin = get_classification_margin(model=target_gcn,
                                               adj=adj,
                                               features=features,
                                               labels=labels,
                                               target_node=target_node)
            cls_margins.append(margin)
        # save number of misclassified target nodes and their classification margins
        print("Saving to %s..." % save_path)
        file_path = os.path.join(save_path, 'margins.npy')
        np.save(file_path, np.array(cls_margins))
        file_path = os.path.join(save_path, 'misclass.npy')
        np.save(file_path, np.array([cnt]))
    else:
        attack_model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False,
                               device=device)

        multi_test_attack(surrogate, target_gcn, attack_model, data,
                          poisoning=args.poisoning,  # poisoning or evasion attack?
                          n_perturbations=args.n_ptb,
                          save_path=save_path,
                          train_iters=args.n_iters)


if __name__ == '__main__':
    main()
