from scipy.optimize import linear_sum_assignment


def make_morph_pairs(dataset):
    c = dataset.generate_cost_matrix(use_labels=True)
    row_ind, col_ind = linear_sum_assignment(c)
    mapping = zip(row_ind, col_ind)
    return dict(mapping)
