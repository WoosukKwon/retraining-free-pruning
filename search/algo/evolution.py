import copy
import random

import numpy as np
from tqdm import tqdm
import torch


class EvolutionFinder:

    def __init__(self, model_config, accuracy_predictor, efficiency_predictor, logger, ranked=False, **kwargs):
        self.config = model_config
        self.accuracy_predictor = accuracy_predictor
        self.efficiency_predictor = efficiency_predictor
        self.logger = logger
        self.ranked = ranked

        self.mutate_prob = kwargs.get('mutate_prob', 0.1)
        self.population_size = kwargs.get('population_size', 100)
        self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

    def random_sample_arch(self, mac_threshold):
        head_prob = torch.ones(self.config.num_attention_heads) * mac_threshold
        filter_prob = torch.ones(self.config.num_filter_groups) * mac_threshold
        head_masks = [torch.bernoulli(head_prob).cuda() for _ in range(self.config.num_hidden_layers)]
        filter_masks = [torch.bernoulli(filter_prob).cuda() for _ in range(self.config.num_hidden_layers)]
        if self.ranked:
            head_masks = [torch.sort(head_mask, descending=True)[0] for head_mask in head_masks]
            filter_masks = [torch.sort(filter_mask, descending=True)[0] for filter_mask in filter_masks]
        return {
            "head_masks": head_masks,
            "filter_masks": filter_masks,
        }

    def random_valid_sample(self, mac_threshold):
        while True:
            sample = self.random_sample_arch(mac_threshold)
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if efficiency <= mac_threshold:
                return sample, efficiency

    def mutate_arch(self, config, mac_threshold):
        head_masks = config["head_masks"]
        filter_masks = config["filter_masks"]
        head_prob = torch.ones(self.config.num_attention_heads) * mac_threshold
        filter_prob = torch.ones(self.config.num_filter_groups) * mac_threshold
        for i in range(self.config.num_hidden_layers):
            if torch.rand(1).item() < self.mutate_prob:
                mask = torch.bernoulli(head_prob).cuda()
                head_masks[i] = torch.sort(mask, descending=True)[0] if self.ranked else mask
            if torch.rand(1).item() < self.mutate_prob:
                mask = torch.bernoulli(filter_prob).cuda()
                filter_masks[i] = torch.sort(mask, descending=True)[0] if self.ranked else mask

    def mutate_sample(self, sample, mac_threshold):
        while True:
            new_sample = copy.deepcopy(sample)
            self.mutate_arch(new_sample, mac_threshold)
            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency <= mac_threshold:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2, mac_threshold):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice(
                        [sample1[key][i], sample2[key][i]])

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency <= mac_threshold:
                return new_sample, efficiency

    @torch.no_grad()
    def search(self, num_iter, mac_threshold):
        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None

        self.logger.info("Generate random population...")
        for _ in range(self.population_size):
            sample, efficiency = self.random_valid_sample(mac_threshold)
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        accs = self.accuracy_predictor.predict_acc(child_pool)
        for i in range(self.population_size):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))

        self.logger.info("Start evolution...")
        with tqdm(total=num_iter, desc='Searching with mac_threshold (%s)' % mac_threshold) as t:
            for i in range(num_iter):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                acc = parents[0][0]
                t.set_postfix({"acc'": parents[0][0]})
                self.logger.info(f"Iter: {i+1} Acc: {parents[0][0]} MAC: {parents[0][2]}")

                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents
                child_pool = []
                efficiency_pool = []

                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    # Mutate
                    new_sample, efficiency = self.mutate_sample(par_sample, mac_threshold)
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)

                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # Crossover
                    new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2, mac_threshold)
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)

                accs = self.accuracy_predictor.predict_acc(child_pool)
                for j in range(self.population_size):
                    population.append((accs[j], child_pool[j], efficiency_pool[j]))

                t.update(1)

        return best_info[1]
