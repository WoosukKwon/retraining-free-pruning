import torch


def all_pairs_dist(matrix):
    num_rows = matrix.shape[0]
    x_sq = matrix.pow(2).sum(dim=1, keepdim=True).expand(num_rows, num_rows)
    dist_sq = x_sq + x_sq.t() - 2 * torch.mm(matrix, matrix.t())
    dist_sq = dist_sq * (dist_sq > 0)
    dist = torch.sqrt(dist_sq)
    return dist
