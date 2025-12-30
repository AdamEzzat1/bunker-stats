pub(crate) fn rolling_mean_std_vec(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    if window == 0 || window > n {
        return (Vec::new(), Vec::new());
    }

    // Kahan-compensated rolling sums (more accurate than naive sum/sumsq, minimal overhead)
    #[inline(always)]
    fn kahan_add(sum: &mut f64, c: &mut f64, x: f64) {
        let y = x - *c;
        let t = *sum + y;
        *c = (t - *sum) - y;
        *sum = t;
    }

    let out_len = n - window + 1;
    let mut means = Vec::with_capacity(out_len);
    let mut stds = Vec::with_capacity(out_len);

    let mut sum = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut sumsq_c = 0.0f64;

    for &x in &xs[..window] {
        kahan_add(&mut sum, &mut sum_c, x);
        kahan_add(&mut sumsq, &mut sumsq_c, x * x);
    }

    let mut push_stats = |sum: f64, sumsq: f64| {
        let mean = sum / window as f64;
        // sample variance (ddof=1), clamp negative due to FP error
        let var = (sumsq - (sum * sum) / window as f64) / ((window - 1) as f64);
        means.push(mean);
        stds.push(var.max(0.0).sqrt());
    };

    push_stats(sum, sumsq);

    for i in window..n {
        let x_new = xs[i];
        let x_old = xs[i - window];
        kahan_add(&mut sum, &mut sum_c, x_new);
        kahan_add(&mut sum, &mut sum_c, -x_old);

        kahan_add(&mut sumsq, &mut sumsq_c, x_new * x_new);
        kahan_add(&mut sumsq, &mut sumsq_c, -(x_old * x_old));

        push_stats(sum, sumsq);
    }

    (means, stds)
}

