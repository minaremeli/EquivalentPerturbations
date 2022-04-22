import numpy as np
from deeprobust.graph.utils import *
from tqdm import tqdm
from utils import equivalent_feature_perturbation
import os
import pickle


def select_nodes(model, data, train_iters=250):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model.initialize()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=train_iters, patience=30)
    model.eval()
    output = model.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[:10]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def get_classification_margin(model, adj, features, labels, target_node):
    model.eval()
    output = model.predict(features, adj)
    margin = classification_margin(output[target_node], labels[target_node])
    return margin


def single_test(adj, features, labels, target_node, gcn, poisoning, idx_train=None, idx_val=None, train_iters=250):
    if poisoning:
        assert idx_train is not None
        assert idx_val is not None
        # test on GCN (poisoning attack)
        gcn.initialize()
        gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=train_iters, patience=30, verbose=True)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        gcn.eval()
        output = gcn.predict(features, adj)

    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()


def multi_test_attack(surrogate, target_gcn, attack_model, data, poisoning, n_perturbations=None, train_iters=250,
                      save_path=None):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if not poisoning:
        target_gcn.initialize()
        target_gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=train_iters, patience=30, verbose=True)

    cnt_s, cnt_f = 0, 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(surrogate, data)
    num = len(node_list)

    s_p_margins = []
    f_p_margins = []
    deltas = {}
    print('=== [%s] Attacking %s nodes respectively ===' % ('Poisoning' if poisoning else 'Evasion', num))
    print(node_list)
    fix_node_perturbations = n_perturbations is not None
    for target_node in tqdm(node_list):
        if not fix_node_perturbations:
            n_perturbations = int(degrees[target_node]) + 2
        attack_model.attack(features, adj, labels, target_node, n_perturbations, verbose=True)
        modified_adj = attack_model.modified_adj
        attack_model.reset()  # reset attack so that it runs correctly on the next target node
        acc = single_test(modified_adj, features, labels, target_node,
                          gcn=target_gcn, poisoning=poisoning,
                          idx_train=idx_train, idx_val=idx_val,
                          train_iters=train_iters)
        if acc == 0:
            cnt_s += 1
        margin_s = get_classification_margin(model=target_gcn,
                                             adj=modified_adj,
                                             features=features,
                                             labels=labels,
                                             target_node=target_node)
        s_p_margins.append(margin_s)

        modified_features, delta = equivalent_feature_perturbation(adj, modified_adj, features, target_node)
        deltas[target_node] = delta
        acc = single_test(adj, modified_features, labels, target_node,
                          gcn=target_gcn, poisoning=poisoning,
                          idx_train=idx_train, idx_val=idx_val,
                          train_iters=train_iters)

        if acc == 0:
            cnt_f += 1
        margin_f = get_classification_margin(model=target_gcn,
                                             adj=adj,
                                             features=modified_features,
                                             labels=labels,
                                             target_node=target_node)
        f_p_margins.append(margin_f)
    print('structure perturbed misclassification rate : %s' % (cnt_s / num))
    print('feature perturbed misclassification rate : %s' % (cnt_f / num))

    print("Structure perturbed classification margins: ", s_p_margins)
    print("Feature perturbed classification margins: ", f_p_margins)

    if save_path is not None:
        print("Saving to %s..." % save_path)
        file_path = os.path.join(save_path, 's_p_margins.npy')
        np.save(file_path, np.array(s_p_margins))
        file_path = os.path.join(save_path, 'f_p_margins.npy')
        np.save(file_path, np.array(f_p_margins))
        file_path = os.path.join(save_path, 's_misclass.npy')
        np.save(file_path, np.array([cnt_s]))
        file_path = os.path.join(save_path, 'f_misclass.npy')
        np.save(file_path, np.array([cnt_f]))
        file_path = os.path.join(save_path, 'delta_dict.pkl')
        pickle.dump(deltas, open(file_path, "wb"))
