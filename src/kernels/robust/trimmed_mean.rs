use crate::mean_slice;

pub(crate) fn trimmed_mean_slice(xs: &[f64], proportion_to_cut: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = v.len();
    let cut = ((n as f64) * proportion_to_cut).floor() as usize;
    if cut * 2 >= n {
        return f64::NAN;
    }
    mean_slice(&v[cut..(n - cut)])
}

