use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::{pyfunction, pymodule};
use pyo3::prelude::{PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;
mod array_reshape;
mod lens;
mod window;


/// A Python module implemented in Rust.
#[pymodule]
fn py_lens(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wave_point, m)?)?;
    //m.add_function(wrap_pyfunction!(wave_line, m)?)?;
    m.add_function(wrap_pyfunction!(star_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(image_to_line, m)?)?;
    m.add_function(wrap_pyfunction!(line_to_image, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_rms_single, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_rms_double, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_rms_triple, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_stdev_single, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_stdev_double, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_stdev_triple, m)?)?;

    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, ctr_x:float, ctr_y:float, u:float, /)")]
    fn wave_point<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        ctr_x: f64,
        ctr_y: f64,
        u: f64,
    ) -> &'py PyArray3<u8> {
        let wp = lens::WavePoint::new((ctr_x, ctr_y), u);
        let img_out = lens::lens_rgb(&py_img.as_array(), wp);
        img_out.to_pyarray(py)
    }
    //#[pyfunction]
    // this is unimplemented
    #[allow(dead_code)]
    fn wave_line<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        ctr_x: f64,
        ctr_y: f64,
        angle: f64,
        u: f64,
    ) -> &'py PyArray3<u8> {
        let wp = lens::WaveLine::new((ctr_x, ctr_y), angle, u);
        let img_out = lens::lens_rgb(&py_img.as_array(), wp);
        img_out.to_pyarray(py)
    }
    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, ctr_x:float, ctr_y:float, point_count:int, u:float, /)")]
    fn star_pattern<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        ctr_x: f64,
        ctr_y: f64,
        point_count: usize,
        u: f64,
    ) -> &'py PyArray3<u8> {
        let wp = lens::StarPattern::new((ctr_x, ctr_y), point_count, u);
        let img_out = lens::lens_rgb(&py_img.as_array(), wp);
        img_out.to_pyarray(py)
    }

    #[pyfunction]
    fn image_to_line<'py>(py: Python<'py>, py_img: PyReadonlyArray3<u8>) -> &'py PyArray2<u8> {
        let image_out = array_reshape::image_to_line(&py_img.as_array());
        image_out.to_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, shape0, shape1 /)")]
    fn line_to_image<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray2<u8>,
        shape0: usize,
        shape1: usize,
    ) -> &'py PyArray3<u8> {
        let image_out = array_reshape::line_to_image(&py_img.as_array(), shape0, shape1);

        image_out.to_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:int, /)")]
    fn windowed_rms_single<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: usize,
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Single(window_size);
        windowed_rms(py, py_img, window_type)
    }

    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:(int,int), /)")]
    fn windowed_rms_double<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: (usize, usize),
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Double(window_size.0, window_size.1);
        windowed_rms(py, py_img, window_type)
    }
    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:(int,int,int), /)")]
    fn windowed_rms_triple<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: (usize, usize, usize),
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Triple(window_size.0, window_size.1, window_size.2);
        windowed_rms(py, py_img, window_type)
    }

    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:int, ddof1:bool, /)")]
    fn windowed_stdev_single<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: usize,
        ddof1: bool,
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Single(window_size);
        windowed_stdev(py, py_img, window_type, ddof1)
    }
    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:(int,int), ddof1:bool, /)")]
    fn windowed_stdev_double<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: (usize, usize),
        ddof1: bool,
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Double(window_size.0, window_size.1);
        windowed_stdev(py, py_img, window_type, ddof1)
    }
    #[pyfunction]
    #[pyo3(text_signature = "(py_img:numpy.ndarray, window_size:(int,int,int), ddof1:bool, /)")]
    fn windowed_stdev_triple<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_size: (usize, usize, usize),
        ddof1: bool,
    ) -> &'py PyArray3<u8> {
        let window_type = window::WindowShape::Triple(window_size.0, window_size.1, window_size.2);
        windowed_stdev(py, py_img, window_type, ddof1)
    }

    fn windowed_rms<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_type: window::WindowShape,
    ) -> &'py PyArray3<u8> {
        let image_out = window::thread_apply_over_window(
            py_img.to_owned_array(),
            window_type,
            window::window_methods::faster_rms_u64_adding,
        );
        image_out.to_pyarray(py)
    }

    fn windowed_stdev<'py>(
        py: Python<'py>,
        py_img: PyReadonlyArray3<u8>,
        window_type: window::WindowShape,
        ddof1: bool,
    ) -> &'py PyArray3<u8> {
        let image_out = window::thread_apply_over_window(
            py_img.to_owned_array(),
            window_type,
            match ddof1 {
                false => window::window_methods::stdev_ddof_0,
                true => window::window_methods::stdev_ddof_1,
            },
        );
        image_out.to_pyarray(py)
    }

    Ok(())
}
