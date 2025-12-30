/// Rolling covariance/correlation over two 1D slices (strict: NaNs propagate).
///
/// These helpers mirror the legacy wrappers in `lib.rs`.
pub(crate) fn rolling_cov_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let x = &x[..n];
    let y = &y[..n];

    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..window {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i >= window {
            let xi_new = x[i];
            let yi_new = y[i];
            let xi_old = x[i - window];
            let yi_old = y[i - window];
            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;
        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);
        out.push(cov);
    }

    out
}

#[allow(dead_code)]
pub(crate) fn rolling_corr_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let x = &x[..n];
    let y = &y[..n];

    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..window {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i >= window {
            let xi_new = x[i];
            let yi_new = y[i];
            let xi_old = x[i - window];
            let yi_old = y[i - window];
            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_x2 += xi_new * xi_new - xi_old * xi_old;
            sum_y2 += yi_new * yi_new - yi_old * yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;

        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);
        let vx = (sum_x2 - w * mx * mx) / ((window - 1) as f64);
        let vy = (sum_y2 - w * my * my) / ((window - 1) as f64);

        let denom = (vx * vy).sqrt();
        out.push(cov / denom);
    }

    out
}