from typing import List

from acyclic_network import AcyclicNetwork
from genome import Genome

import genome_decoder


class GenomeEvaluator:
    def __init__(self, feed_forward_only: bool, evaluator):
        self.feed_forward_only = feed_forward_only
        self.evaluator = evaluator

    def evaluate(self, genome_list: List[Genome]):
        networks: List[AcyclicNetwork] = [genome_decoder.create_acyclic_network(a) for a in genome_list]

        fitnesses = [self.evaluator.evaluate(a) for a in networks]

        for i in range(len(networks)):
            genome_list[i].fitness = fitnesses[i]
            genome_list[i].novelty = 1
