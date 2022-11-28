
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, pymodule, pyfunction};
use pyo3::prelude::{PyModule, PyResult, Python};
use pyo3::{wrap_pyfunction};

mod lens;
mod array_reshape;
mod window;

/// A Python module implemented in Rust.
#[pymodule]
fn py_lens(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WavePoint>()?;
    m.add_class::<WaveLine>()?;
    m.add_function(wrap_pyfunction!(image_to_line, m)?)?;
    m.add_function(wrap_pyfunction!(line_to_image, m)?)?;
    // m.add_function(wrap_pyfunction!(windowed_stdev, m)?)?;
    // #[pyo3(text_signature = "(pt_x0, pt_y0, u, /)")]
    #[pyclass]
    struct WavePoint {
        _tar: lens::WavePoint,
    }
    #[pymethods]
    impl WavePoint {
        #[new]
        fn new(ctr_x: f64, ctr_y: f64, u: f64) -> Self {
            WavePoint {
                _tar: lens::WavePoint::new((ctr_x, ctr_y), u),
            }
        }
        fn apply_wave<'py>(
            &self,
            py_img: PyReadonlyArray3<u8>,
            py: Python<'py>,
        ) -> &'py PyArray3<u8> {
            let image_array = py_img.as_array();
            let image_out = lens::lens_rgb(&image_array, self._tar);
            image_out.to_pyarray(py)
        }
    }

    #[pyclass]
    struct WaveLine {
        _tar: lens::WaveLine,
    }
    #[pymethods]
    impl WaveLine {
        #[new]
        fn new(ctr_x: f64, ctr_y: f64, angle: f64, u: f64) -> Self {
            WaveLine {
                _tar: lens::WaveLine::new((ctr_x, ctr_y), angle, u),
            }
        }
        fn apply_wave<'py>(&self,py: Python<'py>,py_img: PyReadonlyArray3<u8>)
            -> &'py PyArray3<u8> {
            let image_array = py_img.as_array();
            let image_out = lens::lens_rgb(&image_array, self._tar);
            image_out.to_pyarray(py)
        }
    }

    #[pyfunction]
    fn image_to_line<'py>(py:Python<'py>, py_img: PyReadonlyArray3<u8>)-> &'py PyArray2<u8>{
        let image_out = array_reshape::image_to_line(&py_img.as_array());
        image_out.to_pyarray(py)

    }

    #[pyfunction]
    fn line_to_image<'py>(py:Python<'py>, py_img: PyReadonlyArray2<u8>, shape0:usize, shape1:usize)
        -> &'py PyArray3<u8> {
        let image_out = array_reshape::line_to_image(
            &py_img.as_array(), shape0, shape1
        );

        image_out.to_pyarray(py)
    }


    #[pyfunction]
    fn windowed_stdev<'py>(py:Python<'py>, py_img: PyReadonlyArray3<u8>, window_size:usize)-> &'py PyArray3<u8>{
        let image_out = window::thread_apply_over_window(
            py_img.to_owned_array(),
            window::WindowShape::Single(window_size),
            window::window_apply_methods::mystd
        );
        image_out.to_pyarray(py)
    }


    Ok(())
}
