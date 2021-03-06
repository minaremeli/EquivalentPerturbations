import argparse
import torch
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset, PtbDataset
from deeprobust.graph.global_attack import Metattack
from os import path
import os
from utils import equivalent_feature_perturbation

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, help='Random seeds.')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--n_hidden', type=int, default=16, help='Dimension of node embeddings.')
parser.add_argument('--n_iters', type=int, default=200, help='Number of training iterations.')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--results_dir', type=str, default='.', help='Path of directory where results should be saved.')

# datasets for which have ready-made attacked graphs
pre_attacked_datasets = ['cora', 'citeseer', 'polblogs']

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = path.join(args.results_dir,
                      "mettack",
                      args.dataset)
if not path.exists(save_path):
    os.makedirs(save_path)
print("Results will be saved at: ", save_path)


def set_seed(seed):
    print("Seed set to ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)


def main():
    clean_accs = []
    s_p_accs = []
    f_p_accs = []
    for seed in args.seeds:
        set_seed(seed)  # set seed

        data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_unlabeled = np.union1d(idx_val, idx_test)

        target_gcn = GCN(nfeat=features.shape[1],
                         nhid=args.n_hidden,
                         nclass=labels.max().item() + 1,
                         lr=0.01,
                         with_relu=True,
                         with_bias=False,
                         device=device)
        target_gcn = target_gcn.to(device)

        target_gcn.initialize()
        target_gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=args.n_iters, patience=30, verbose=True)
        target_gcn.eval()

        # clean accuracy
        test_acc = target_gcn.test(idx_test)
        clean_accs.append(test_acc)


        if args.dataset in pre_attacked_datasets and args.ptb_rate == 0.05:
            print('==================')
            print('=== load graph perturbed by DeepRobust 5% metattack (under prognn splits) ===')
            perturbed_data = PtbDataset(root='/tmp/',
                                        name=args.dataset,
                                        attack_method='meta')
            perturbed_adj = perturbed_data.adj
        else:
            print('==================')
            print('=== create perturbed graph with %d%% metattack ===' % (int(args.ptb_rate * 100)))
            surrogate = GCN(nfeat=features.shape[1],
                            nhid=args.n_hidden,
                            nclass=labels.max().item() + 1,
                            dropout=0,
                            lr=0.01,
                            weight_decay=0,
                            with_relu=False,
                            with_bias=False,
                            device=device)
            surrogate = surrogate.to(device)
            surrogate.fit(features, adj, labels, idx_train)

            perturbations = int(args.ptb_rate * (adj.sum() // 2))
            attack_model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=0)
            attack_model = attack_model.to(device)
            attack_model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
            perturbed_adj = attack_model.modified_adj

        target_gcn.initialize()
        target_gcn.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=args.n_iters, patience=30, verbose=True)
        target_gcn.eval()

        # structure poisoned accuracy
        s_p_test_acc = target_gcn.test(idx_test)
        s_p_accs.append(s_p_test_acc)

        print('==================')
        print('=== turn structure perturbation into feature perturbation ===')
        modified_features, delta = equivalent_feature_perturbation(adj, perturbed_adj, features)

        target_gcn.initialize()
        target_gcn.fit(modified_features, adj, labels, idx_train, idx_val, train_iters=args.n_iters, patience=30, verbose=True)
        target_gcn.eval()

        # feature poisoned accuracy
        f_p_test_acc = target_gcn.test(idx_test)
        f_p_accs.append(f_p_test_acc)

    # save accuracies
    file_path = os.path.join(save_path, 'test_accs.npy')
    np.save(file_path, np.array(clean_accs))
    file_path = os.path.join(save_path, 's_p_test_accs.npy')
    np.save(file_path, np.array(s_p_accs))
    file_path = os.path.join(save_path, 'f_p_test_accs.npy')
    np.save(file_path, np.array(f_p_accs))


if __name__ == '__main__':
    main()