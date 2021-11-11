import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def partition_dataset(dataset, ratios):
    num_partition = len(ratios)
    num_data = len(dataset)
    indicies = list(range(num_data))

    ratios = np.array(ratios)
    num_each_partition = ratios * num_data

    partitioned = []
    start_idx = 0
    for i in range(num_partition):
        end_idx = start_idx + int(num_each_partition[i]) if i != num_partition - 1 else num_data
        subset_sampler = SubsetRandomSampler(indicies[start_idx:end_idx])
        partitioned.append(subset_sampler)
        start_idx = end_idx

    return tuple(partitioned)
