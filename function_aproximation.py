import numpy
from joblib import Parallel

import evolution_parameters
import genome_decoder
import genome_evaluator
import activation_function_library
import xml_parser
from acyclic_network import AcyclicNetwork
from evolution_algorithm import EvolutionAlgorithm
from genome_factory import GenomeFactory
from kmeans_clustering import KMeansClustering
from manhattan_distance_metric import ManhattanDistanceMetric
import activation_fns
from neat_type import NeatType
import matplotlib.pyplot as plt


def fn(x):
    return activation_fns.sine(x * 2)


def main():
    evol_params = evolution_parameters.default_evolution_parameters
    kmeans = KMeansClustering(ManhattanDistanceMetric())
    activation_fn_library = activation_function_library.default_library
    genome_factory = GenomeFactory(1, 1, activation_fn_library)

    fn_evaluator = FunctionEvaluator(fn)
    list_evaluator = genome_evaluator.GenomeEvaluator(True, fn_evaluator)

    ea = EvolutionAlgorithm(evol_params, kmeans, NeatType.Objective, list_evaluator, genome_factory, 100)
    ea.initialization()

    while True:
        with Parallel(n_jobs=14, backend="threading") as parallel:
            ea.perform_generation(parallel)
            if ea.current_champ is not None:
                print("Generation " + str(ea.genome_factory.current_generation))
                print(ea.current_champ.fitness)

                if ea.genome_factory.current_generation % 100 == 0:
                    for i in range(len(ea.genome_list)):

                        output_values = []
                        example_values = []
                        network = genome_decoder.create_acyclic_network(ea.genome_list[i])

                        for x, y in fn_evaluator.sample_data:
                            network.set_input([x])
                            network.activate()
                            out = network.get_output()[0]
                            output_values.append(out)
                            example_values.append(y)

                        plt.clf()
                        plt.plot(output_values)
                        plt.plot(example_values, "r-.")
                        plt.savefig("sine" + str(i) + ".png")
                        plt.clf()

                plt.clf()
                plt.plot([a.fitness for a in ea.genome_list])
                plt.savefig("fitnesses.png")


class FunctionEvaluator:
    def __init__(self, fn):
        self.fn = fn
        self.sample_data = [((a / 100.0), fn(a / 100.0)) for a in range(101)]

    def evaluate(self, network: AcyclicNetwork) -> float:
        error = 0.0

        for data in self.sample_data:
            x, y = data
            network.set_input([x])
            network.activate()
            out = network.get_output()

            error += (y - out[0]) ** 2

        error = error / len(self.sample_data)

        if error > 1:
            return 0.00000001

        return 1.0 - numpy.sqrt(error)


if __name__ == '__main__':
    main()
