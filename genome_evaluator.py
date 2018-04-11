from typing import List
from joblib import Parallel, delayed

from acyclic_network import AcyclicNetwork
from genome import Genome

import genome_decoder

def evaluate(a) -> float:
    genome, evaluator = a
    return evaluator(genome)

class GenomeEvaluator:
    def __init__(self, feed_forward_only: bool, evaluator):
        self.feed_forward_only = feed_forward_only
        self.evaluator = evaluator

    def evaluate(self, genome_list: List[Genome], parallel):
        networks: List[AcyclicNetwork] = parallel(delayed(genome_decoder.create_acyclic_network)(a) for a in genome_list)
        fitnesses = parallel(delayed(evaluate)(i) for i in [(a, self.evaluator.evaluate) for a in networks])

        for i in range(len(networks)):
            genome_list[i].fitness = fitnesses[i]
            genome_list[i].novelty = 1
