use std::time::Instant;

use ndarray;
use ndarray::Array3;
use numpy;
use numpy::PyArray3;
use numpy::PyReadonlyArray3;
use numpy::ToPyArray;
use pyo3::IntoPy;
use pyo3::prelude::pymodule;
use pyo3::prelude::PyModule;
use pyo3::prelude::PyObject;
use pyo3::prelude::PyResult;
use pyo3::prelude::Python;

#[warn(unused_imports)]
mod lens;

/// A Python module implemented in Rust.
#[pymodule]
fn py_lens(_py: Python, _m: &PyModule) -> PyResult<()> {
    #[pyfn(_m)]
    fn apply_lens<'py>(py: Python<'py>, py_img: PyReadonlyArray3<u8>, cx: f32, cy: f32, u: f32) -> &'py PyArray3<u8> {
        let image_array = py_img.as_array();
        let image_out = lens::apply_lens_rgb(&image_array, cx, cy, u);
        let py_array = image_out.to_pyarray(py);
        py_array
    }

    #[pyfn(_m)]
    fn in_rust_test(_py: Python<'_>) {
        let (_w, _h, _d): (usize, usize, usize) = (1024, 1024, 3);
        let an_array: Array3<u8> = Array3::ones((_w, _h, _d));
        let cx: f32 = 0.3;
        let cy: f32 = 0.3;
        let u: f32 = 15.0;

        let now = Instant::now();
        let _v0 = lens::apply_lens_rgb(&an_array.view(), cx, cy, u);
        println!("lens Timed: ([{} ,{} ,{} ,u8], {} ,{} ,{}) : {}",
                _w, _h, _d,
                cx,cy,u,
                now.elapsed().as_secs_f32()
        );
    }

    #[pyfn(_m)]
    fn space(_py: Python<'_>, x: f32, y: f32, cx: f32, cy: f32, u: f32) -> PyObject {
        let var = lens::space_def(x, y, cx, cy, u).to_vec();
        var.into_py(_py)
    }

    Ok(())
}
