from deeprobust.graph import utils
import time
import numpy as np
import scipy.sparse as sp


def equivalent_feature_perturbation(adj, adj_m, features, target_node=None):
    start = time.time()
    modified_features = features.tolil().copy()

    adj_norm = utils.normalize_adj(adj.tocsr())
    adj_norm = adj_norm.dot(adj_norm)
    adj_m_norm = utils.normalize_adj(adj_m.tocsr())
    adj_m_norm = adj_m_norm.dot(adj_m_norm)

    D_ii = sp.diags(adj_norm.diagonal()).power(-1)  # element-wise power, not inverse!

    adj_diff = adj_m_norm - adj_norm

    delta = D_ii.dot(adj_diff.dot(features))
    end = time.time()
    print("Converted structure perturbations to feature perturbations in %f ms" % (end - start))

    if target_node is not None:
        # update only for target node
        modified_features[target_node] += delta[target_node]
        return modified_features.tocsr(), delta[target_node]
    else:
        # update for all nodes
        modified_features += delta
        return modified_features.tocsr(), delta

