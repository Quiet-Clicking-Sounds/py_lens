import numpy


def wave_point(py_img: numpy.ndarray, ctr_x: float, ctr_y: float, u: float) -> numpy.ndarray:
    """

    :param py_img: image as array, expects shape() = (x,y,3)
    :param ctr_x: pixel centre position
    :param ctr_y: pixel centre position
    :param u:
    """
    ...

# not implemented
# def wave_line(py_img: numpy.ndarray, ctr_x: float, ctr_y: float, angle: float, u: float) -> numpy.ndarray:
#     """
#     :param py_img: image as array, expects shape() = (x,y,3)
#     :param ctr_x: pixel centre position
#     :param ctr_y: pixel centre position
#     :param angle: angle in radians
#     :param u:
#     """
#     ...

def star_pattern(py_img: numpy.ndarray, ctr_x: float, ctr_y: float, point_count: int, u: float) -> numpy.ndarray:
    """
    :param py_img: image as array, expects shape() = (x,y,3)
    :param ctr_x: pixel centre position
    :param ctr_y: pixel centre position
    :param point_count: number of points in the pattern
    :param u:
    """
    ...


def image_to_line(py_img: numpy.ndarray) -> numpy.ndarray:
    """
    :param py_img: 3d numpy array
    :return: 2d numpy array,
    """
    ...


def line_to_image(py_img: numpy.ndarray, shape0: int, shape1: int) -> numpy.ndarray:
    """
    :param shape1:  shape of output image (should be the same as what was put into image_to_line)
    :param shape0:  shape of output image (should be the same as what was put into image_to_line)
    :param py_img: 2d numpy array
    :return: 3d numpy array,
    """
    ...


def windowed_rms_single(py_img: numpy.ndarray, window_size: int, ddof1: bool) -> numpy.ndarray:
    """
    rms function over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w,w,1] to traverse the array
    :param ddof1: if true set ddof value to 1, else ddof is 0
    :return: numpy array with dimensions of the input[x,y,z] - [w-1, w-1, 0]
    """


def windowed_rms_double(py_img: numpy.ndarray, window_size: tuple[int, int], ddof1: bool) -> numpy.ndarray:
    """
    rms function over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w0,w1,1] to traverse the array
    :param ddof1: if true set ddof value to 1, else ddof is 0
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, 0]
    """


def windowed_rms_triple(py_img: numpy.ndarray, window_size: tuple[int, int, int], ddof1: bool) -> numpy.ndarray:
    """
    rms function over a windowed array,

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w0,w1,w2] to traverse the array
    :param ddof1: if true set ddof value to 1, else ddof is 0
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, w2-1]
    """

def windowed_stdev_single(py_img: numpy.ndarray, window_size: int) -> numpy.ndarray:
    """
    standard deviation over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w,w,1] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w-1, w-1, 0]
    """


def windowed_stdev_double(py_img: numpy.ndarray, window_size: tuple[int, int]) -> numpy.ndarray:
    """
    standard deviation over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w0,w1,1] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, 0]
    """


def windowed_stdev_triple(py_img: numpy.ndarray, window_size: tuple[int, int, int]) -> numpy.ndarray:
    """
    standard deviation over a windowed array,

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moving window of shape [w0,w1,w2] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, w2-1]
    """
