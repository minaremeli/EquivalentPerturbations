from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from utils import equivalent_feature_perturbation

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--n_ptb', type=int, default=0,
                    help='Number of perturbations. If not set, will default to node degree + 2.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

# Setup Attack Model
target_node = 0
assert target_node in idx_unlabeled

model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
model = model.to(device)


def test_equivalent_feature_perturbations():
    degrees = adj.sum(0).A1
    if args.n_ptb == 0:
        # How many perturbations to perform. Default: Degree of the node + 2
        n_perturbations = int(degrees[target_node]) + 2
    else:
        n_perturbations = args.n_ptb

    print("Target node degree: ", int(degrees[target_node]))

    # direct attack
    model.attack(features, adj, labels, target_node, n_perturbations)

    # # indirect attack/ influencer attack
    # model.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5)
    modified_adj = model.modified_adj
    modified_features, _ = equivalent_feature_perturbation(target_node, adj, modified_adj, features)
    print(model.structure_perturbations)

    print('=== testing GCN on original(clean) graph ===')
    test(surrogate, adj, features, target_node)
    print('=== testing GCN on structure perturbed graph ===')
    test(surrogate, modified_adj, features, target_node)
    print('=== testing GCN on feature perturbed graph ===')
    test(surrogate, adj, modified_features, target_node)


def test(gcn, adj, features, target_node):
    ''' test on GCN '''

    gcn.eval()
    output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])[0]
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


if __name__ == '__main__':
    test_equivalent_feature_perturbations()
