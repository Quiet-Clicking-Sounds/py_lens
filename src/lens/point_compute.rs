use std::collections::HashMap;
use num_traits::{abs, FloatConst};

fn apply_rt_sgn_sq(hyp: f64) -> f64 {
    ((hyp.abs()).sqrt()).sin().powi(2).copysign(hyp)
}
pub trait ComputePoint {
    fn setup_for_new_image(&mut self, x: usize, y: usize);
    fn point_shift(&self, x: f64, y: f64) -> (f64, f64);
}

#[derive(Copy, Clone)]
struct Scale {
    _m: f64,
    _d: f64,
}

impl Scale {
    ///
    /// Scale is generated as a unit distance between opposing corners of the image
    ///
    /// # Arguments
    ///
    /// * `x`: image width
    /// * `y`: image height
    /// * `u`: scale multiplier
    ///
    /// returns: Scale
    ///
    pub fn new(x: usize, y: usize, u: f64) -> Self {
        let (x, y) = (x as f64, y as f64);
        let xy = x.hypot(y) * u;
        Scale {
            _m: xy,
            _d: xy.recip(),
        }
    }
    pub fn default() -> Self {
        Scale { _m: 1f64, _d: 1f64 }
    }
    ///
    /// Returns (x * scale)
    ///
    pub fn mul(&self, x: f64) -> f64 {
        x * self._m
    }
    ///
    /// Returns (x / scale)  (uses inverse multiplication)
    ///
    pub fn div(&self, x: f64) -> f64 {
        x * self._d
    }
}


#[derive(Copy, Clone)]
pub struct StarPattern{
    ctr: (f64, f64),
    ctr_x: f64,
    ctr_y: f64,
    pts: usize,
    pt_ang:f64,
    s: Scale,
    u: f64,
}

impl ComputePoint for StarPattern{

    fn setup_for_new_image(&mut self, x: usize, y: usize) {
        self.s = Scale::new(x, y, self.u);
        self.ctr_x = self.ctr.0 * if (0f64 <= self.ctr.0 )&( self.ctr.0 <= 1f64)
        { x as f64} else { 1f64};
        self.ctr_y = self.ctr.1 * if (0f64 <= self.ctr.1 )&( self.ctr.1 <= 1f64)
        { y as f64} else { 1f64};

    }

    fn point_shift(&self, x: f64, y: f64) -> (f64, f64) {
        let xa = x - self.ctr_x;
        let ya = y - self.ctr_y;

        let atan = ya.atan2(xa);
        let sico = atan.sin_cos();
        let qfactor  = (atan.div_euclid(self.pt_ang) - self.pt_ang).abs() * self.u;
        let ang = (self.pt_ang).sin_cos();

        (
            qfactor * sico.1 * ang.1 + x, // this is not correct
            qfactor * sico.0 * ang.0 + y // this is not correct
        )

    }
}

impl StarPattern {
    pub fn new(ctr: (f64, f64), pts:usize, u: f64) -> Self {
        StarPattern {
            ctr,
            ctr_x: ctr.0,
            ctr_y: ctr.1,
            s: Scale::default(),
            pts,
            pt_ang:f64::PI()/pts as f64,
            u,
        }
    }
}

#[derive(Copy, Clone)]
pub struct WaveLine {
    ctr: (f64, f64),
    ctr_x: f64,
    ctr_y: f64,
    _a_cos: f64,
    _a_cos_90: f64,
    _a_sin: f64,
    _a_sin_90: f64,
    s: Scale,
    u: f64,
}

impl WaveLine {
    pub fn new(c0: (f64, f64), angle: f64, u: f64) -> Self {
        let offset_angle = std::f64::consts::PI / 2.0;
        let _a_cos_90 = (angle.cos() - offset_angle).rem_euclid(std::f64::consts::PI * 2.0);
        let _a_sin_90 = (angle.sin() - offset_angle).rem_euclid(std::f64::consts::PI * 2.0);
        WaveLine {
            ctr:c0,
            ctr_x: c0.0,
            ctr_y: c0.1,
            _a_cos: angle.cos(),
            _a_cos_90,
            _a_sin: angle.sin(),
            _a_sin_90,
            s: Scale::default(),
            u,
        }
    }
}

impl ComputePoint for WaveLine {
    fn setup_for_new_image(&mut self, x: usize, y: usize) {
        self.s = Scale::new(x, y, self.u);
        self.ctr_x = self.ctr.0 * if (0f64 <= self.ctr.0 )&( self.ctr.0 <= 1f64)
        { x as f64} else { 1f64};
        self.ctr_y = self.ctr.1 * if (0f64 <= self.ctr.1 )&( self.ctr.1 <= 1f64)
        { y as f64} else { 1f64};
    }
    fn point_shift(&self, x: f64, y: f64) -> (f64, f64) {
        let hyp = self._a_cos * (self.ctr_y - y) - self._a_sin * (self.ctr_x - x);

        let hyp = apply_rt_sgn_sq(hyp);

        let hyp = self.s.mul(hyp);
        (hyp * self._a_cos_90 + x, hyp * self._a_sin_90 + y)
    }
}

#[derive(Copy, Clone)]
pub struct WavePoint {
    ctr: (f64, f64),
    ctr_x: f64,
    ctr_y: f64,
    s: Scale,
    u: f64,
}

impl WavePoint {
    pub fn new(ctr: (f64, f64), u: f64) -> Self {
        WavePoint {
            ctr,
            ctr_x: ctr.0,
            ctr_y: ctr.1,
            s: Scale::default(),
            u,
        }
    }
}

impl ComputePoint for WavePoint {
    fn setup_for_new_image(&mut self, x: usize, y: usize) {
        self.s = Scale::new(x, y, self.u);
        self.ctr_x = self.ctr.0 * if (0f64 <= self.ctr.0 )&( self.ctr.0 <= 1f64)
        { x as f64} else { 1f64};
        self.ctr_y = self.ctr.1 * if (0f64 <= self.ctr.1 )&( self.ctr.1 <= 1f64)
        { y as f64} else { 1f64};
    }
    fn point_shift(&self, x: f64, y: f64) -> (f64, f64) {
        let xa = x - self.ctr_x;
        let ya = y - self.ctr_y;
        let ang: (f64, f64) = ya.atan2(xa).sin_cos();

        let hyp = xa.hypot(ya);
        // let hyp = self.s.div(hyp);
        let hyp = apply_rt_sgn_sq(hyp);
        let hyp = self.s.mul(hyp);

        (hyp * ang.1 + x, hyp * ang.0 + y)
    }
}

unsafe impl Send for Scale {}
unsafe impl Sync for Scale {}
unsafe impl Send for WaveLine {}
unsafe impl Sync for WaveLine {}
unsafe impl Send for WavePoint {}
unsafe impl Sync for WavePoint {}
