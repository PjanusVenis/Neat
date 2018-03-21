import random

import numpy

import activation_fns
import genome_parameters
from activation_function_library import ActivationFnLibrary
from genome import Genome
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
        self.neuron_innovation_number = 0
        self.connection_innovation_number = 0
        self.genome_list = []
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

        self.possible_io_connections_ids = [(x.inno_id, y.inno_id, self.next_connection_innovation())
                                            for x in self.input_neurons for y in self.output_neurons]

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

    def create_genome(self, birth_generation: int):
        num_connections = max(1, self.genome_params.initial_interconnections_proportion * len(self.possible_io_connections_ids))
        connections = {ConnectionGene(i, a, b, self.random_weight()) for (a, b, i)
                       in numpy.random.choice(self.possible_io_connections_ids, num_connections)}

        neurons = {a.inno_id: NeuronGene(a.inno_id, a.activation_fn, a.type) for a in self.basic_neurons}

        for conn in connections:
            source_neuron = neurons[conn.from_id]
            target_neuron = neurons[conn.to_id]

            source_neuron.target_neurons.append(target_neuron)
            target_neuron.source_neurons.append(source_neuron)

        return Genome(neurons, connections, birth_generation)

    def create_offspring(self, genome: Genome) -> Genome:
        new_genome = Genome(genome.neuron_gene_list, genome.connection_gene_list, self.current_generation)
        return new_genome

    def mutate_genome(self, genome: Genome):
        success = False
        while not success:
            success = numpy.random.choice(self.mutations, p = self.mutation_probabilities)(genome)

    def mutate_weights(self, genome: Genome) -> bool:
        return

    def mutate_add_node(self, genome: Genome) -> bool:
        if len(genome.connection_gene_list) == 0:
            return False

        connection_to_replace = numpy.random.choice(genome.connection_gene_list)
        genome.connection_gene_list.remove(connection_to_replace)
        neuron_id = 0

        if connection_to_replace.inno_id in self.connection_replaced:
            neuron_id = self.connection_replaced[connection_to_replace.inno_id]
        else:
            neuron_id = self.next_neuron_innovation()
            self.connection_replaced[connection_to_replace.inno_id] = neuron_id

        new_neuron = NeuronGene(neuron_id, self.activation_fn_library.get_random_function(), NeuronType.HIDDEN)
        new_connection1 = self.create_connection(connection_to_replace.from_id, neuron_id, connection_to_replace.weight)
        new_connection2 = self.create_connection(neuron_id, connection_to_replace.to_id, 1.0)

        new_neuron.target_neurons.add(genome.neuron_gene_list[connection_to_replace.to_id])
        new_neuron.source_neurons.add(genome.neuron_gene_list[connection_to_replace.from_id])

        genome.neuron_gene_list[new_neuron.inno_id] = new_neuron
        genome.connection_gene_list.add(new_connection1)
        genome.connection_gene_list.add(new_connection2)

        return True

    def create_connection(self, from_id: int, to_id: int, weight: float) -> ConnectionGene:
        conn_id = self.connection_innovation[(from_id, to_id)] if (from_id, to_id) in self.connection_innovation \
            else self.next_connection_innovation()
        return ConnectionGene(conn_id, from_id, to_id, weight)


    def mutate_add_connection(self, genome: Genome) -> bool:
        return

    def mutate_delete_connection(self, genome: Genome) -> bool:
        return


