import numpy




class WaveLine:
    def __init__(self, ctr_x: float, ctr_y: float, angle: float, u: float):
        """

        :param ctr_x: pixel centre position
        :param ctr_y: pixel centre position
        :param angle: angle in radians
        :param u:
        """
        ...
    def apply_wave(self, py_img: numpy.ndarray):
        """
        :param py_img: image as array, expects shape() = (x,y,3)
        """
        ...
class WavePoint:
    def __init__(self, ctr_x:float, ctr_y:float, u:float):
        """

        :param ctr_x: pixel centre position
        :param ctr_y: pixel centre position
        :param u:
        """
        ...
    def apply_wave(self, py_img: numpy.ndarray):
        """
        :param py_img: image as array, expects shape() = (x,y,3)
        """
        ...

def image_to_line(py_img:numpy.ndarray)-> numpy.ndarray:
    """
    :param py_img: 3d numpy array
    :return: 2d numpy array,
    """
    ...
def line_to_image(py_img:numpy.ndarray, shape0:int, shape1:int)-> numpy.ndarray:
    """
    :param shape1:  shape of output image (should be the same as what was put into image_to_line)
    :param shape0:  shape of output image (should be the same as what was put into image_to_line)
    :param py_img: 2d numpy array
    :return: 3d numpy array,
    """
    ...


def windowed_stdev_single(py_img: numpy.ndarray, window_size: int) -> numpy.ndarray:
    """
    standard deviation over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moiving window of shape [w,w,1] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w-1, w-1, 0]
    """
def windowed_stdev_double(py_img:numpy.ndarray, window_size:tuple[int,int])-> numpy.ndarray:
    """
    standard deviation over a windowed array, based on image processing, ignores the 3rd layer

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moiving window of shape [w0,w1,1] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, 0]
    """
def windowed_stdev_triple(py_img:numpy.ndarray, window_size:tuple[int,int,int])-> numpy.ndarray:
    """
    standard deviation over a windowed array,

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param window_size: window size, this will create a moiving window of shape [w0,w1,w2] to traverse the array
    :return: numpy array with dimensions of the input[x,y,z] - [w0-1, w1-1, w2-1]
    """
