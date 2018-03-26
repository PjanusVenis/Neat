from typing import List

from genome import Genome
from specie import Specie


class KMeansClustering:
    def __init__(self, distance_metric):
        self.max_loops = 5
        self.distance_metric = distance_metric

    def speciate_genomes(self, genome_list: List[Genome], specie_count: int) -> List[Specie]:
        specie_list: List[Specie] = []
        for i in range(specie_count):
            new_specie = Specie()
            selected_genome = genome_list[i]
            selected_genome.specie = new_specie
            new_specie.centroid = selected_genome.get_position()
            specie_list.append(new_specie)

        for i in range(specie_list, len(genome_list)):
            selected_genome = genome_list[i]
            closest_specie = self.find_closest_specie(selected_genome, specie_list)
            selected_genome.specie = closest_specie
            closest_specie.genome_list.append(selected_genome)

        for specie in specie_list:
            specie.centroid = self.calculate_specie_centroid(specie)

        return self.speciate_until_convergence(specie_list)

    def speciate_offsprings(self, offspring_list: List[Genome], specie_list: List[Specie]) -> List[Specie]:
        for specie in specie_list:
            specie.centroid = self.calculate_specie_centroid(specie)

        for offspring in offspring_list:
            closest_specie = self.find_closest_specie(offspring, specie_list)
            closest_specie.genome_list.append(offspring)
            offspring.specie = closest_specie

        for specie in specie_list:
            specie.centroid = self.calculate_specie_centroid(specie)

        return self.speciate_until_convergence(specie_list)

    def speciate_until_convergence(self, specie_list: List[Specie]) -> List[Specie]:
        reallocations = 0

        genome_list: List[Genome] = sum([a.genome_list for a in specie_list], [])

        for step in range(self.max_loops):
            closest_species = [self.find_closest_specie(a) for a in genome_list]

            for i in range(len(genome_list)):
                if closest_species[i] != genome_list[i].specie:
                    genome = genome_list[i]
                    specie = specie_list[i]
                    genome.specie.genome_list.remove(genome)
                    genome.specie = specie
                    specie.genome_list.append(genome)
                    reallocations += 1

            if reallocations == 0:
                break

            for specie in specie_list:
                if len(specie.genome_list) == 0:
                    genomes_by_distance = self.get_genomes_by_distance_from_species(genome_list)
                    selected_genome = genomes_by_distance[-1]
                    original_specie = selected_genome.specie
                    original_specie.genome_list.remove(selected_genome)
                    specie.genome_list.append(selected_genome)

                    original_specie.centroid = self.calculate_specie_centroid(original_specie)
                    specie.centroid = self.calculate_specie_centroid(specie)

            for specie in specie_list:
                specie.centroid = self.calculate_specie_centroid(specie)

        return specie_list

    def calculate_specie_centroid(self, specie: Specie) -> List[(int, float)]:
        return self.distance_metric.calculate_centroid([a.get_position() for a in specie.genome_list])

    def get_genomes_by_distance_from_species(self, genome_list: List[Genome]) -> List[Genome]:
        distance_list = [(self.distance_metric.measure_distance(a.get_position(), a.specie.centroid, a)) for a in genome_list]
        distance_list.sort(key=lambda x: x[0])
        return [a for _, a in distance_list]

    def find_closest_specie(self, genome: Genome, specie_list: List[Specie]) -> Specie:
        return min([(self.distance_metric.measure_distance(genome.get_position(), b.centroid), b) for b in specie_list],
                   key=lambda x: x[0])[1]
