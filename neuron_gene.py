from neuron_type import NeuronType


class NeuronGene:
    def __init__(self, inno_id: int, activation_fn, type: NeuronType):
        self.inno_id = inno_id
        self.activation_fn = activation_fn
        self.type = type
        self.target_neurons: {NeuronGene} = set()
        self.source_neurons: {NeuronGene} = set()

    def is_redundant(self):
        return type == NeuronType.HIDDEN and (len(self.target_neurons) == 0 or len(self.source_neurons))