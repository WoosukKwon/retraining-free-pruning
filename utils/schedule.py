import math


def get_pruning_schedule(target, num_iter):
    p = math.pow(target, 1 / num_iter)
    schedule = [p ** i for i in range(1, num_iter)] + [target]
    return schedule
