import evolution_parameters
import genome_evaluator
import genome_parameters
import activation_function_library
from acyclic_network import AcyclicNetwork
from evolution_algorithm import EvolutionAlgorithm
from genome_factory import GenomeFactory
from kmeans_clustering import KMeansClustering
from manhattan_distance_metric import ManhattanDistanceMetric
import activation_fns
from neat_type import NeatType


def main():
    evol_params = evolution_parameters.default_evolution_parameters
    kmeans = KMeansClustering(ManhattanDistanceMetric())
    activation_fn_library = activation_function_library.default_library
    genome_factory = GenomeFactory(1, 1, activation_fn_library)
    fn_evaluator = FunctionEvaluator(activation_fns.sine)
    list_evaluator = genome_evaluator.GenomeEvaluator(True, fn_evaluator)

    ea = EvolutionAlgorithm(evol_params, kmeans, NeatType.Objective, list_evaluator, genome_factory, 500)

    while True:
        ea.perform_generation()
        if ea.current_champ is not None:
            print("Generation")
            print(ea.current_champs[5].fitness)


class FunctionEvaluator:
    def __init__(self, fn):
        self.fn = fn
        self.sample_data = [(a / 100.0, fn(a / 100.0)) for a in range(100)]

    def evaluate(self, network: AcyclicNetwork) -> float:
        error = 0.0

        for data in self.sample_data:
            x, y = data
            network.set_input([x])
            network.activate()
            out = network.get_output()
            error += abs(y - out[0])

        error = error / len(self.sample_data)
        return (1.0 - error) ** 2

if __name__ == '__main__':
    main()