import random
from typing import List, Tuple

import numpy
import time

import activation_fns
import genome_parameters
from activation_function_library import ActivationFnLibrary
from correlation_item_type import CorrelationItemType
from correlation_statistics import CorrelationStatistics
from genome import Genome
from neat_type import NeatType
from neuron_gene import NeuronGene
from connection_gene import ConnectionGene
from neuron_type import NeuronType


class GenomeFactory:
    def __init__(self, input_count: int, output_count: int, activation_fn_library: ActivationFnLibrary,
                 genome_params=genome_parameters.default_genome_parameters):
        self.input_count = input_count
        self.output_count = output_count
        self.activation_fn_library = activation_fn_library
        self.genome_params = genome_params
        self.connection_innovation = {}
        self.neuron_innovation = {}
        self.neuron_innovation_number = 0
        self.connection_innovation_number = 0
        self.current_generation = 0

        self.connection_replaced = {}

        self.input_neurons = []
        self.output_neurons = []
        self.bias_neuron = NeuronGene(self.next_neuron_innovation(), activation_fns.identity, NeuronType.BIAS)

        self.mutations = [self.mutate_weights, self.mutate_add_node, self.mutate_add_connection, self.mutate_delete_connection]
        self.mutation_probabilities = [genome_params.connection_weight_probability, genome_params.add_node_probability,
                                       genome_params.add_connection_probability, genome_params.delete_connection_probability]

        for i in range(self.input_count):
            self.input_neurons.append(NeuronGene(self.next_neuron_innovation(), activation_fns.identity, NeuronType.INPUT))

        for i in range(self.output_count):
            self.output_neurons.append(NeuronGene(self.next_neuron_innovation(), activation_fns.identity, NeuronType.OUTPUT))

        self.basic_neurons = [self.bias_neuron] + self.input_neurons + self.output_neurons

        self.possible_io_connections_ids = {(x.inno_id, y.inno_id)
                                            for x in self.input_neurons for y in self.output_neurons}

    def next_neuron_innovation(self) -> int:
        innovation = self.neuron_innovation_number
        self.neuron_innovation_number += 1
        return innovation

    def next_connection_innovation(self) -> int:
        innovation = self.connection_innovation_number
        self.connection_innovation_number += 1
        return innovation

    def random_weight(self) -> float:
        return random.uniform(-self.genome_params.connection_weight_range, self.genome_params.connection_weight_range)

    def create_genome_list(self, length: int, birth_generation: int) -> list:
        return [self.create_genome(birth_generation) for _ in range(length)]

    def create_blank_neuron(self) -> Genome:
        neuron_list = [NeuronGene(a.inno_id, a.activation_fn, a.type) for a in self.basic_neurons]
        neurons = {a.inno_id: a for a in neuron_list}

        return Genome(neurons, neuron_list, [], self.current_generation)

    def create_genome(self, birth_generation: int):
        num_connections = max(1, self.genome_params.initial_interconnections_proportion * len(self.possible_io_connections_ids))
        connections = [self.create_connection(a, b, self.random_weight()) for (a, b)
                       in random.sample(self.possible_io_connections_ids, int(num_connections))]

        neuron_list = [NeuronGene(a.inno_id, a.activation_fn, a.type) for a in self.basic_neurons]
        neurons = {a.inno_id: a for a in neuron_list}

        bias = neurons[0]

        for conn in connections:
            source_neuron = neurons[conn.from_id]
            target_neuron = neurons[conn.to_id]

            source_neuron.target_neurons.append(target_neuron)
            target_neuron.source_neurons.append(source_neuron)

        for neuron in neuron_list:
            if neuron == bias:
                continue

            bias.target_neurons.append(neuron)
            neuron.source_neurons.append(bias)

            connection = self.create_connection(bias.inno_id, neuron.inno_id, self.random_weight())
            connections.append(connection)

        return Genome(neurons, neuron_list, connections, birth_generation)

    def create_offspring(self, genome: Genome) -> Genome:
        neuron_gene_list = []
        neuron_gene_dict = {}

        for i in range(len(genome.neuron_gene_list)):
            original_gene = genome.neuron_gene_list[i]
            neuron_gene = NeuronGene(original_gene.inno_id, original_gene.activation_fn, original_gene.type)

            neuron_gene_list.append(neuron_gene)
            neuron_gene_dict[neuron_gene.inno_id] = neuron_gene

        connection_genes = []

        for conn in genome.connection_gene_list:
            connection_genes.append(ConnectionGene(conn.inno_id, conn.from_id, conn.to_id, conn.weight))
            from_neuron = neuron_gene_dict[conn.from_id]
            to_neuron = neuron_gene_dict[conn.to_id]
            from_neuron.target_neurons.append(to_neuron)
            to_neuron.source_neurons.append(from_neuron)

        new_genome = Genome(neuron_gene_dict, neuron_gene_list, connection_genes, self.current_generation)

        self.mutate_genome(new_genome)

        return new_genome

    def create_offspring_sexual(self, parent1: Genome, parent2: Genome, neat_type: NeatType) -> Genome:
        correlation_list, correlation_stats = \
            self.correlate_connection_lists(parent1.connection_gene_list, parent2.connection_gene_list)

        offspring = self.create_blank_neuron()
        offspring.last_mutation = "Sexual"

        fitness_switch = False
        if neat_type == NeatType.Novelty:
            if parent1.novelty > parent2.novelty:
                fitness_switch = True
            elif parent2.novelty == parent1.novelty:
                fitness_switch = random.random() < 0.5

        elif neat_type == NeatType.CNAOS or neat_type == NeatType.Objective:
            if parent1.fitness > parent2.fitness:
                fitness_switch = True
            elif parent2.fitness == parent1.fitness:
                fitness_switch = random.random() < 0.5

        combine_disjoint_excess = random.random() < self.genome_params.disjoint_excess_recombine_probability
        disjoint_excess_list: List[(ConnectionGene, ConnectionGene, CorrelationItemType)] = []

        for conn1, conn2, corr_type in correlation_list:
            selection_switch = False

            if corr_type == CorrelationItemType.Match:
                selection_switch = random.random() < 0.5
            elif fitness_switch and conn1 is not None:
                selection_switch = True
            elif not fitness_switch and conn2 is not None:
                selection_switch = False
            else:
                if combine_disjoint_excess:
                    disjoint_excess_list.append((conn1, conn2, corr_type))
                continue

            if selection_switch:
                connection = conn1
            else:
                connection = conn2

            self.add_connection(offspring, connection, corr_type == CorrelationItemType.Match)
            if (self.genome_params.feed_forward_only and not offspring.is_connection_cyclic(connection.from_id, connection.to_id)) or \
                    not self.genome_params.feed_forward_only:
                offspring.connection_gene_list.remove(connection)
                from_neuron = offspring.neuron_gene_dict[connection.from_id]
                to_neuron = offspring.neuron_gene_dict[connection.to_id]

                from_neuron.target_neurons.remove(to_neuron)
                to_neuron.source_neurons.remove(from_neuron)

        if combine_disjoint_excess:
            for conn1, conn2, corr_type in disjoint_excess_list:
                if conn1 is None:
                    connection = conn2
                else:
                    connection = conn1

                self.add_connection(offspring, connection, False)

                if (self.genome_params.feed_forward_only and not offspring.is_connection_cyclic(connection.from_id, connection.to_id)) or \
                        not self.genome_params.feed_forward_only:
                    offspring.connection_gene_list.remove(connection)
                    from_neuron = offspring.neuron_gene_dict[connection.from_id]
                    to_neuron = offspring.neuron_gene_dict[connection.to_id]

                    from_neuron.target_neurons.remove(to_neuron)
                    to_neuron.source_neurons.remove(from_neuron)

        return offspring

    def add_connection(self, genome: Genome, conn: ConnectionGene, overwrite_existing: bool):
        connection_genes_dict = {a.inno_id: a for a in genome.connection_gene_list}

        if conn.inno_id in connection_genes_dict:
            if overwrite_existing:
                genome.connection_gene_list[conn.inno_id].weight = conn.weight
        else:
            genome.connection_gene_list.append(conn)
            added_neuron = False

            if conn.from_id not in genome.neuron_gene_dict:
                neuron = NeuronGene(conn.from_id, self.neuron_innovation[conn.from_id].activation_fn, self.neuron_innovation[conn.from_id].type)
                genome.neuron_gene_dict[neuron.inno_id] = neuron
                genome.neuron_gene_list.append(neuron)

                added_neuron = True
            if conn.to_id not in genome.neuron_gene_dict:
                neuron = NeuronGene(conn.to_id, self.neuron_innovation[conn.to_id].activation_fn, self.neuron_innovation[conn.to_id].type)
                genome.neuron_gene_dict[neuron.inno_id] = neuron
                genome.neuron_gene_list.append(neuron)

                added_neuron = True

            from_neuron = genome.neuron_gene_dict[conn.from_id]
            to_neuron = genome.neuron_gene_dict[conn.to_id]

            from_neuron.target_neurons.append(to_neuron)
            to_neuron.source_neurons.append(from_neuron)

            genome.connection_gene_list = sorted(genome.connection_gene_list, key=lambda x: x.inno_id)
            if added_neuron:
                genome.neuron_gene_list = sorted(genome.neuron_gene_list, key=lambda x: x.inno_id)

    def correlate_connection_lists(self, conns1: List[ConnectionGene], conns2: List[ConnectionGene]) -> (List[Tuple[ConnectionGene, ConnectionGene, CorrelationItemType]],
                                                                                                         CorrelationStatistics):
        correlation_stats: CorrelationStatistics = CorrelationStatistics()
        correlation_list: List[(ConnectionGene, ConnectionGene, CorrelationItemType)] = []

        if len(conns1) == 0:
            correlation_stats.excess_gene_count = len(conns2)
            for gene in conns2:
                correlation_list.append((None, gene, CorrelationItemType.Excess))
            return correlation_list, correlation_stats
        elif len(conns2) == 0:
            correlation_stats.excess_gene_count = len(conns1)
            for gene in conns1:
                correlation_list.append((None, gene, CorrelationItemType.Excess))
            return correlation_list, correlation_stats

        conn1_idx = 0
        conn2_idx = 0

        while True:
            gene1 = conns1[conn1_idx]
            gene2 = conns2[conn2_idx]

            if gene2.inno_id < gene1.inno_id:
                correlation_list.append((None, gene2, CorrelationItemType.Disjoint))
                correlation_stats.disjoint_gene_count += 1

                conn2_idx += 1
            elif gene2.inno_id == gene1.inno_id:
                correlation_list.append((gene1, gene2, CorrelationItemType.Match))
                correlation_stats.matching_gene_count += 1

                conn2_idx += 1
                conn1_idx += 1
            else:
                correlation_list.append((gene1, None, CorrelationItemType.Excess))
                correlation_stats.excess_gene_count += 1

                conn1_idx += 1

            if conn1_idx == len(conns1):
                correlation_stats.excess_gene_count += len(conns2) - (conn2_idx + 1)
                correlation_list.extend([(None, a, CorrelationItemType.Excess) for a in conns2[conn2_idx:]])

                return correlation_list, correlation_stats

            if conn2_idx == len(conns2):
                correlation_stats.disjoint_gene_count += len(conns1) - (conn1_idx + 1)
                correlation_list.extend([(a, None, CorrelationItemType.Disjoint) for a in conns1[conn1_idx:]])

                return correlation_list, correlation_stats

    def mutate_genome(self, genome: Genome):
        success = False
        while not success:
            success = numpy.random.choice(self.mutations, p=self.mutation_probabilities)(genome)

    def mutate_weights(self, genome: Genome) -> bool:
        num_connection_mutation: float = numpy.sin(random.random() * (numpy.pi / 2)) * len(genome.connection_gene_list)
        connections_to_mutate: List[ConnectionGene] = random.sample(genome.connection_gene_list, int(num_connection_mutation))
        for conn in connections_to_mutate:
                conn.weight = numpy.clip(conn.weight + (random.random() - 0.5),
                                         -self.genome_params.connection_weight_range, self.genome_params.connection_weight_range)

        genome.last_mutation = "Mutate weights"

        return num_connection_mutation > 0

    def mutate_add_node(self, genome: Genome) -> bool:
        if len(genome.connection_gene_list) == 0:
            return False

        genome.last_mutation = "Add node"

        connection_to_replace = numpy.random.choice(genome.connection_gene_list)
        genome.connection_gene_list.remove(connection_to_replace)
        neuron_id = 0

        if connection_to_replace.inno_id in self.connection_replaced:
            neuron_id = self.connection_replaced[connection_to_replace.inno_id]
        else:
            neuron_id = self.next_neuron_innovation()
            self.connection_replaced[connection_to_replace.inno_id] = neuron_id

        new_neuron = NeuronGene(neuron_id, self.activation_fn_library.get_random_function(), NeuronType.HIDDEN)

        self.neuron_innovation[new_neuron.inno_id] = new_neuron

        new_connection1 = self.create_connection(connection_to_replace.from_id, neuron_id, connection_to_replace.weight)
        new_connection2 = self.create_connection(neuron_id, connection_to_replace.to_id, 1.0)

        from_neuron = genome.neuron_gene_dict[connection_to_replace.from_id]
        to_neuron = genome.neuron_gene_dict[connection_to_replace.to_id]

        from_neuron.target_neurons.remove(to_neuron)
        to_neuron.source_neurons.remove(from_neuron)

        new_neuron.target_neurons.append(genome.neuron_gene_dict[connection_to_replace.to_id])
        new_neuron.source_neurons.append(genome.neuron_gene_dict[connection_to_replace.from_id])

        genome.neuron_gene_dict[new_neuron.inno_id] = new_neuron
        genome.neuron_gene_list.append(new_neuron)

        from_neuron.target_neurons.append(new_neuron)
        to_neuron.source_neurons.append(new_neuron)

        genome.connection_gene_list.append(new_connection1)
        genome.connection_gene_list.append(new_connection2)

        return True

    def create_connection(self, from_id: int, to_id: int, weight: float) -> ConnectionGene:
        if (from_id, to_id) in self.connection_innovation:
            conn_id = self.connection_innovation[(from_id, to_id)]
        else:
            conn_id = self.next_connection_innovation()
            self.connection_innovation[(from_id, to_id)] = conn_id

        return ConnectionGene(conn_id, from_id, to_id, weight)

    def mutate_add_connection(self, genome: Genome) -> bool:
        neuron_count = len(genome.neuron_gene_list)
        hidden_output_neuron_count = neuron_count - self.input_count
        input_bias_hidden_neuron_count = neuron_count - self.output_count

        genome.last_mutation = "Add"

        if self.genome_params.feed_forward_only:
            for attempts in range(5):
                source_neuron_idx = random.randint(0, input_bias_hidden_neuron_count - 1)
                if self.input_count + self.output_count > source_neuron_idx >= self.input_count:
                    source_neuron_idx += self.output_count

                target_neuron_idx = self.input_count + random.randint(0, hidden_output_neuron_count - 1)
                if source_neuron_idx == target_neuron_idx:
                    target_neuron_idx += 1
                    if target_neuron_idx == neuron_count:
                        continue

                source_neuron = genome.neuron_gene_list[source_neuron_idx]
                target_neuron = genome.neuron_gene_list[target_neuron_idx]

                if target_neuron.type == NeuronType.BIAS or target_neuron in source_neuron.target_neurons or genome.is_connection_cyclic(source_neuron.inno_id, target_neuron.inno_id):
                    continue

                self.mutate_add_create_connection(genome, source_neuron, target_neuron)
                return True
        else:
            for attempts in range(5):
                source_neuron_idx = random.randint(0, neuron_count - 1)
                target_neuron_idx = self.input_count + random.randint(0, hidden_output_neuron_count - 1)

                source_neuron = genome.neuron_gene_list[source_neuron_idx]
                target_neuron = genome.neuron_gene_list[target_neuron_idx]

                self.mutate_add_create_connection(genome, source_neuron, target_neuron)
                return True

    def mutate_add_create_connection(self, genome: Genome, source_neuron: NeuronGene, target_neuron: NeuronGene):
        connection = self.create_connection(source_neuron.inno_id, target_neuron.inno_id, self.random_weight())
        genome.connection_gene_list.append(connection)
        source_neuron.target_neurons.append(target_neuron)
        target_neuron.source_neurons.append(source_neuron)

    def mutate_delete_connection(self, genome: Genome) -> bool:
        if len(genome.connection_gene_list) < 2:
            return False

        connection_to_delete: ConnectionGene = random.choice(genome.connection_gene_list)
        from_neuron = genome.neuron_gene_dict[connection_to_delete.from_id]
        to_neuron = genome.neuron_gene_dict[connection_to_delete.to_id]

        from_neuron.target_neurons.remove(to_neuron)
        to_neuron.source_neurons.remove(from_neuron)

        genome.connection_gene_list.remove(connection_to_delete)

        genome.last_mutation = "Delete"

        return True


def test():
    start = time.time()
    a = ActivationFnLibrary(activation_fns.sine, activation_fns.gaussian)
    b = GenomeFactory(5, 5, a)
    c = b.create_genome_list(5000, 9)
    map(lambda x: b.mutate_genome(x), c)
    print(time.time() - start)
