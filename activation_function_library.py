from typing import List

import numpy

import activation_fns


class ActivationFnLibrary:
    def __init__(self, fns: list, probabilities: List[float]):
        self.fns = fns
        self.probabilities = probabilities

    def get_random_function(self):
        return numpy.random.choice(self.fns, p=self.probabilities)


default_library = ActivationFnLibrary([activation_fns.binary, activation_fns.gaussian, activation_fns.identity,
                                       activation_fns.relu, activation_fns.sigmoid, activation_fns.sine, activation_fns.selu],
                                      [1.0 / 7.0 for _ in range(7)])