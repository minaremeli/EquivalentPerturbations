from deeprobust.graph import utils
import time
import numpy as np


def equivalent_feature_perturbation(target_node, adj, adj_m, features):
    start = time.time()
    modified_features = features.tolil().copy()

    adj_norm = utils.normalize_adj(adj.tocsr())
    adj_norm = adj_norm.dot(adj_norm)
    adj_m_norm = utils.normalize_adj(adj_m.tocsr())
    adj_m_norm = adj_m_norm.dot(adj_m_norm)

    adj_norm_ii_inv = np.power(adj_norm[target_node, target_node], -1)

    adj_diff = adj_m_norm - adj_norm

    delta = adj_diff.dot(features) * adj_norm_ii_inv
    modified_features[target_node] += delta[target_node]
    end = time.time()
    print("Converted structure perturbations to feature perturbations in %f ms" % (end - start))
    return modified_features.tocsr(), delta[target_node]


