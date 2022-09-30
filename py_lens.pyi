import numpy


def apply_lens(py_img: numpy.ndarray, cx: float, cy: float, u: float) -> numpy.ndarray:
    """
    :param py_img: input image, shape must be [x size, y size, 3],
    :type py_img: numpy.ndarray
    :param cx: centre position of the wave in x - positions 0<x<1 will be accepted as (dimension / x)
    :type cx: float
    :param cy: centre position of the wave in y - positions 0<x<1 will be accepted as (dimension / x)
    :type cy: float
    :param u: wave intensity
    :type u: float
    :return: image with wave function applied
    :rtype: numpy.ndarray
    """
    ...


def in_rust_test():
    """
    prints time taken to compute a default lens - for testing purposes.
    :return:
    :rtype:
    """
    ...


def space(x, y, cx, cy, u) -> list[float]:
    """
    test function for the spacial coordinates,

    :param x: x coordinate
    :type x: float
    :param y: y coordinate
    :type y: float
    :param cx: centre position of the wave in x
    :type cx: float
    :param cy: centre position of the wave in y
    :type cy: float
    :param u: wave intensity
    :type u: float
    :return: new [x,y] position in list format
    :rtype: list[float]
    """
