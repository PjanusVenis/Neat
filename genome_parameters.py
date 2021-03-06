class GenomeParameters:
    def __init__(self, connection_weight_range: float, initial_interconnections_proportion: float,
                 disjoint_excess_recombine_probability: float, connection_weight_probability: float,
                 add_node_probability: float, add_connection_probability: float,
                 delete_connection_probability: float, feed_forward_only: bool):
        self.connection_weight_range = connection_weight_range
        self.initial_interconnections_proportion = initial_interconnections_proportion
        self.disjoint_excess_recombine_probability = disjoint_excess_recombine_probability
        self.connection_weight_probability = connection_weight_probability
        self.add_node_probability = add_node_probability
        self.add_connection_probability = add_connection_probability
        self.delete_connection_probability = delete_connection_probability
        self.feed_forward_only = feed_forward_only


default_genome_parameters = GenomeParameters(5.0, 0.05, 0.1, 0.988, 0.002, 0.01, 0.000, True)
