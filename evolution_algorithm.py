import random
from typing import List

import numpy

from evolution_parameters import EvolutionParameters
from genome import Genome
from genome_factory import GenomeFactory
from kmeans_clustering import KMeansClustering
from neat_type import NeatType
from specie_stats import SpecieStats


class EvolutionAlgorithm:
    def __init__(self, evolution_params: EvolutionParameters, speciation_strategy: KMeansClustering, neat_type: NeatType,
                 genome_list_evaluator, genome_factory: GenomeFactory, population_size: int):
        self.evolution_params = evolution_params
        self.speciation_strategy = speciation_strategy
        self.neat_type = neat_type
        self.genome_list_evaluator = genome_list_evaluator
        self.genome_factory = genome_factory
        self.population_size = population_size
        self.genome_list = genome_factory.create_genome_list(population_size, 0)
        self.current_champ: Genome = None
        self.current_champs: List[Genome] = []

        self.genome_list_evaluator.evaluate(self.genome_list)
        self.specie_list = self.speciation_strategy.speciate_genomes(self.genome_list, evolution_params.specie_count)
        self.sort_specie_genomes()
        self.update_best_genome()

    def perform_generation(self):
        return

    def calculate_specie_stats(self):
        total_mean_fitness = 0.0
        specie_count = len(self.specie_list)
        specie_stats_list: List[SpecieStats] = []

        for specie in self.specie_list:
            specie_stats = SpecieStats()
            if self.neat_type == NeatType.CNAOS:
                specie_stats.mean_fitness = specie.mean_novelty() * specie.mean_fitness()
            elif self.neat_type == NeatType.Novelty:
                specie_stats.mean_fitness = specie.mean_novelty()
            elif self.neat_type == NeatType.Objective:
                specie_stats.mean_fitness = specie.mean_fitness()
            total_mean_fitness += specie_stats.mean_fitness
            specie_stats_list.append(specie_stats)

        error = 0.0
        total_target_size_int = 0
        if total_mean_fitness == 0:
            target_size_real = self.population_size / specie_count
            for specie_stats in specie_stats_list:
                target_size_int = probabilistic_round(target_size_real)
                specie_stats.target_size_real = target_size_real
                specie_stats.target_size_int = target_size_int
                total_target_size_int += target_size_int
        else:
            for specie_stats in specie_stats_list:
                specie_stats.target_size_real = (specie_stats.mean_fitness / total_mean_fitness) * self.population_size
                error += specie_stats.target_size_real - int(specie_stats.target_size_real)
                if error >= 1.0:
                    specie_stats.target_size_int = int(specie_stats.target_size_real) + 1
                    error -= 1
                else:
                    specie_stats.target_size_int = int(specie_stats.target_size_real)
                total_target_size_int += specie_stats.target_size_int

            if round(error) != 0.0:
                random.choice(specie_stats).target_size_int += 1
                total_target_size_int += 1

        offspring_count = 0

        for i in range(specie_count):
            specie_stats = specie_stats_list[i]

            if specie_stats.target_size_int == 0:
                specie_stats.elite_size_int = 0
                continue

            elite_size_real = len(self.specie_list[i].genome_list) * self.evolution_params.elitism_proportion
            elite_size_int = probabilistic_round(elite_size_real)

            specie_stats.elite_size_int = min(elite_size_int, specie_stats.target_size_int)
            specie_stats.offspring_count = specie_stats.target_size_int - specie_stats.elite_size_int
            offspring_count += specie_stats.offspring_count

            offspring_asexual_count = specie_stats.offspring_count * self.evolution_params.offspring_asexual_proportion
            specie_stats.offspring_asexual_count = probabilistic_round(offspring_asexual_count)
            specie_stats.offspring_sexual_count = specie_stats.offspring_count - specie_stats.offspring_asexual_count

            selection_size = len(self.specie_list[i].genome_list) * self.evolution_params.selection_proportion
            specie_stats.selection_size_int = max(1, probabilistic_round(selection_size))

        return specie_stats_list

    def create_offspring(self, specie_stats_list: List[SpecieStats], offspring_count: int):
        specie_count = len(self.specie_list)
        specie_fitness_arr: List[float] = []
        probabilities: List[List[float]] = []

        for i in range(specie_count):
            specie_stats = specie_stats_list[i]
            specie_fitness_arr.append(specie_stats.selection_size_int)
            probabilities.append([])

            for j in range(specie_stats.selection_size_int):
                if self.neat_type == NeatType.Objective or self.neat_type == NeatType.CNAOS:
                    probabilities[i][j] = self.specie_list[i].genome_list[j].fitness
                elif self.neat_type == NeatType.Novelty:
                    probabilities[i][j] = self.specie_list[i].genome_list[j].novelty

        offspring_list: List[Genome] = []

        for specie_idx in range(specie_count):
            specie_stats = specie_stats_list[specie_idx]
            specie_probabilities = probabilities[specie_idx]

            specie_offsprings = [self.genome_factory.create_offspring(a) for a in numpy.random.choice(self.specie_list[specie_idx].genome_list, specie_stats.offspring_asexual_count,
                                                                p=specie_probabilities)]

            cross_specie_mating_count = probabilistic_round(self.evolution_params.offspring_interspecies_proportion * specie_stats.offspring_sexual_count)

            mating_count = 0

            cross_specie_offsprings = [ for _ in range(cross_specie_mating_count)]



    def sort_specie_genomes(self):
        for specie in self.specie_list:
            specie.genome_list.sort(key=self.creator_comparator())

    def update_best_genome(self):
        self.current_champs = [max(a.genome_list, key=lambda x: x.fitness) for a in self.specie_list]
        self.current_champ = max(self.current_champs, key=lambda x: x.fitness)

    def creator_comparator(self):
        def comparator(a: Genome, b: Genome):
            if self.neat_type == NeatType.CNAOS or self.neat_type == NeatType.Objective:
                metric_a = a.fitness
                metric_b = b.fitness
            elif self.neat_type == NeatType.Novelty:
                metric_a = a.novelty
                metric_b = b.novelty

            if metric_a > metric_b:
                return -1
            elif metric_b > metric_a:
                return 1

            if a.birth_generation > b.birth_generation:
                return -1
            elif b.birth_generation > a.birth_generation:
                return 1

            return 0

def probabilistic_round(a: float) -> int:
    integer_part = int(a)
    fractional_part = a - integer_part
    return integer_part + 1 if random.random() < fractional_part else integer_part