import numpy
from joblib import Parallel

import evolution_parameters
import genome_decoder
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
import matplotlib.pyplot as plt
import gym

from neuron_gene import NeuronGene


def main():
    evol_params = evolution_parameters.default_evolution_parameters
    kmeans = KMeansClustering(ManhattanDistanceMetric())
    activation_fn_library = activation_function_library.default_library
    genome_factory = GenomeFactory(4, 2, activation_fn_library)

    cart_pole = CartPole()
    list_evaluator = genome_evaluator.GenomeEvaluator(True, cart_pole)

    ea = EvolutionAlgorithm(evol_params, kmeans, NeatType.Objective, list_evaluator, genome_factory, 150)
    ea.initialization()

    with Parallel(n_jobs=4, backend="threading") as parallel:
        while True:
            ea.perform_generation(parallel)
            if ea.current_champ is not None:
                print("Generation " + str(ea.genome_factory.current_generation))
                print(ea.current_champ.fitness)

                if ea.genome_factory.current_generation % 100 == 0:
                    network = genome_decoder.create_acyclic_network(ea.current_champ)

                    observation = cart_pole.env.reset()

                    while True:
                        cart_pole.env.render()
                        observation[0] /= 2.4
                        observation[2] /= 41.8

                        network.set_input(observation)
                        network.activate()
                        output = network.get_output()[0]
                        action = 0 if output > 0 else 1

                        observation, reward, done, info = cart_pole.env.step(action)

                        if done:
                            break


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def evaluate(self, network: AcyclicNetwork) -> float:
        fitness = 0
        observation = self.env.reset()

        while True:
            observation[0] /= 2.4
            observation[2] /= 41.8
            network.set_input(observation)
            network.activate()
            output = network.get_output()[0]
            action = 0 if output > 0 else 1

            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break

        return fitness


if __name__ == '__main__':
    main()