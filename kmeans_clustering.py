from typing import List

from genome import Genome
from specie import Specie


class KMeansClustering:
    def __init__(self, distance_metric):
        self.max_loops = 5
        self.distance_metric = distance_metric

    def speciate_genomes(self, genome_list: List[Genome], specie_count: int) -> List[Specie]:
        return

    def find_closest_specie(self, genome: Genome, specie_list: List[Specie]) -> Specie:
        closest_specie = specie_list[0]
        shortest_distance = self.distance_metric.me
