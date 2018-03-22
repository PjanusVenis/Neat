import numpy


def identity(a: float) -> float:
    return a


def sigmoid(a: float) -> float:
    return 1 / (1 + numpy.exp(-a))


def relu(a: float) -> float:
    return 0 if a < 0 else a


def sine(a: float) -> float:
    return numpy.sin(a)


def binary(a: float) -> float:
    return 0 if a < 0 else 1


def gaussian(a: float) -> float:
    return numpy.exp(-(a**2))
