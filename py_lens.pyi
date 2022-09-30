import numpy

def apply_lens(py_img: numpy.ndarray, cx: float, cy: float, u: float) -> numpy.ndarray:
    """
    :param py_img: input image, shape must be [x size, y size, 3],
    :param cx: centre location of the wave function in x - positions 0<cx<1 will be treated as a fraction of the total image width
    :param cy: centre location of the wave function in y - positions 0<cy<1 will be treated as a fraction of the total image height
    :param u: wave intensity
    :return: image with wave function applied
    """
    ...

def in_rust_test():
    """
    prints time taken to compute a default lens - for testing purposes.
    :return: None
    """
    ...

def space(x: float, y: float, cx: float, cy: float, u: float) -> list[float]:
    """
    test function for the spacial coordinate function

    :param x: x coordinate
    :param y: y coordinate
    :param cx: centre position of the wave in x
    :param cy: centre position of the wave in y
    :param u: wave intensity
    :return: new list[x,y] position
    """
    ...
