from typing import List

import numpy


class ActivationFnLibrary:
    def __init__(self, fns: list, probabilities: List[float]):
        self.fns = fns
        self.probabilities = probabilities

    def get_random_function(self):
        return numpy.random.choice(self.fns, p = self.probabilities)