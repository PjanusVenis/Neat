from typing import List

import activation_fns
from activation_function_library import ActivationFnLibrary
from connection_gene import ConnectionGene
from genome import Genome

import xml.etree.ElementTree as ET

from genome_factory import GenomeFactory
from neuron_gene import NeuronGene
from neuron_type import NeuronType


def name_to_fn(name: str):
    if name.lower() == "linear":
        return activation_fns.identity
    elif name.lower() == "bipolarsigmoid":
        return activation_fns.sigmoid
    elif name.lower() == "gaussian":
        return  activation_fns.gaussian
    elif name.lower() == "sine":
        return activation_fns.sine
    elif name.lower() == "selu":
        return  activation_fns.selu
    elif name.lower() == "step":
        return activation_fns.binary

    print("activation fn not found")
    return


def name_to_type(name: str) -> NeuronType:
        name = name.lower()

        if name == "bias":
            return NeuronType.BIAS
        elif name == "input":
            return NeuronType.INPUT
        elif name == "output":
            return NeuronType.OUTPUT
        elif name == "hidden":
            return  NeuronType.HIDDEN

        print("type not found")
        return None


def read_generation_xml(filename: str, genome_factory: GenomeFactory) -> List[Genome]:
    tree = ET.parse(filename)
    root = tree.getroot()

    fns = []
    total_probability = 0

    for a in root.iter("ActivationFunctions"):
        for b in a:
            fn_xml = b.attrib
            fn = name_to_fn(fn_xml["Name"])
            prob = float(fn_xml["Probability"])
            total_probability += prob
            fns.append((fn, prob))

    fns = [(a, b / total_probability) for a, b in fns]

    library = ActivationFnLibrary([a for a, _ in fns], [a for _, a in fns])

    genomes = []

    for networks in root.iter("Networks"):
        for network in networks.iter("Network"):
            neurons = []
            neuron_dict = {}
            connections = []

            for nodes in network.iter("Nodes"):
                for node in nodes.iter("Node"):
                    node_type = name_to_type(node.attrib["Type"])
                    n_id = int(node.attrib["Id"])
                    fn_id = int(node.attrib["ActivationId"])

                    neuron = NeuronGene(n_id, library.fns[fn_id], node_type)
                    genome_factory.neuron_innovation[n_id] = neuron
                    neurons.append(neuron)

            neuron_dict = {a.inno_id: a for a in neurons}

            for conns in network.iter("Connections"):
                for conn in conns.iter("Connection"):
                    conn_id = int(conn.attrib["Id"])
                    source_id = int(conn.attrib["SourceId"])
                    tgt_id = int(conn.attrib["TargetId"])
                    weight = float(conn.attrib["Weight"])

                    connection = ConnectionGene(conn_id, source_id, tgt_id, weight)

                    from_neuron = neuron_dict[source_id]
                    to_neuron = neuron_dict[tgt_id]

                    from_neuron.target_neurons.append(to_neuron)
                    to_neuron.source_neurons.append(from_neuron)

                    genome_factory.connection_innovation[(source_id, tgt_id)] = conn_id

                    connections.append(connection)

            genome = Genome(neuron_dict, neurons, connections, 0)
            genomes.append(genome)

        genome_factory.activation_fn_library = library
        return genomes

