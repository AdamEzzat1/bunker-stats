// ========================================================================
// Multiple testing p-value adjustment
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use super::common::reject_nonfinite;

/// Adjust p-values for multiple testing.
///
/// Methods:
///   - "bonferroni": p_adj = min(p * n, 1)
///   - "holm": step-down Bonferroni (Holm 1979)
///   - "bh" / "fdr_bh": Benjamini-Hochberg FDR control
///
/// Returns adjusted p-values in the same order as input.
#[pyfunction]
#[pyo3(signature = (pvalues, method="bh"))]
pub fn p_adjust_np<'py>(
    py: Python<'py>,
    pvalues: PyReadonlyArray1<f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pvals = pvalues.as_slice()?;
    let n = pvals.len();

    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    reject_nonfinite(pvals, "pvalues")?;

    // Validate: all p-values must be in [0, 1]
    for &p in pvals {
        if p < 0.0 || p > 1.0 {
            return Err(PyValueError::new_err(
                "p-values must be in [0, 1]",
            ));
        }
    }

    let adjusted = match method.to_lowercase().as_str() {
        "bonferroni" => bonferroni(pvals),
        "holm" => holm(pvals),
        "bh" | "fdr_bh" | "benjamini-hochberg" => benjamini_hochberg(pvals),
        _ => {
            return Err(PyValueError::new_err(
                "method must be one of: 'bonferroni', 'holm', 'bh'",
            ))
        }
    };

    Ok(PyArray1::from_vec_bound(py, adjusted))
}

/// Bonferroni correction: p_adj = min(p * n, 1.0)
fn bonferroni(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len() as f64;
    pvals.iter().map(|&p| (p * n).min(1.0)).collect()
}

/// Holm step-down procedure (Holm 1979)
///
/// 1. Sort p-values ascending.
/// 2. p_adj[i] = max(p_adj[i-1], p[i] * (n - i))
/// 3. Return to original order.
fn holm(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();

    // Create index-value pairs and sort by p-value
    let mut indexed: Vec<(usize, f64)> = pvals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut running_max = 0.0f64;

    for (rank, &(orig_idx, p)) in indexed.iter().enumerate() {
        let multiplier = (n - rank) as f64;
        let adj = (p * multiplier).min(1.0);
        running_max = running_max.max(adj);
        adjusted[orig_idx] = running_max;
    }

    adjusted
}

/// Benjamini-Hochberg FDR control
///
/// 1. Sort p-values descending.
/// 2. p_adj[i] = min(p_adj[i+1], p[i] * n / rank)
/// 3. Return to original order.
fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();

    // Create index-value pairs and sort by p-value descending
    let mut indexed: Vec<(usize, f64)> = pvals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut running_min = 1.0f64;

    for (i, &(orig_idx, p)) in indexed.iter().enumerate() {
        // rank = n - i (since sorted descending, rank in ascending order is n-i)
        let rank = (n - i) as f64;
        let adj = (p * (n as f64) / rank).min(1.0);
        running_min = running_min.min(adj);
        adjusted[orig_idx] = running_min;
    }

    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let adj = bonferroni(&pvals);
        assert!((adj[0] - 0.04).abs() < 1e-12);
        assert!((adj[1] - 0.16).abs() < 1e-12);
        assert!((adj[2] - 0.12).abs() < 1e-12);
        assert!((adj[3] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_holm() {
        // p = [0.01, 0.04, 0.03, 0.005]
        // sorted: [0.005, 0.01, 0.03, 0.04]  ranks: 1,2,3,4
        // adj_sorted: [0.005*4, max(0.02, 0.01*3), max(0.03, 0.03*2), max(0.06, 0.04*1)]
        //           = [0.02,    0.03,              0.06,              0.06]
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let adj = holm(&pvals);
        assert!((adj[3] - 0.02).abs() < 1e-12);   // 0.005 * 4
        assert!((adj[0] - 0.03).abs() < 1e-12);   // max(0.02, 0.01*3)
        assert!((adj[2] - 0.06).abs() < 1e-12);   // max(0.03, 0.03*2)
        assert!((adj[1] - 0.06).abs() < 1e-12);   // max(0.06, 0.04*1) => monotonic enforcement
    }

    #[test]
    fn test_bh() {
        // p = [0.01, 0.04, 0.03, 0.005]
        // sorted ascending: [0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1)]
        // BH from descending: sorted desc = [0.04(idx1), 0.03(idx2), 0.01(idx0), 0.005(idx3)]
        // adj[idx1] = min(1, 0.04*4/4) = 0.04
        // adj[idx2] = min(0.04, 0.03*4/3) = 0.04
        // adj[idx0] = min(0.04, 0.01*4/2) = 0.02
        // adj[idx3] = min(0.02, 0.005*4/1) = 0.02
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let adj = benjamini_hochberg(&pvals);
        assert!((adj[0] - 0.02).abs() < 1e-12);
        assert!((adj[1] - 0.04).abs() < 1e-12);
        assert!((adj[2] - 0.04).abs() < 1e-12);
        assert!((adj[3] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_bonferroni_clamping() {
        let pvals = vec![0.5, 0.6, 0.8];
        let adj = bonferroni(&pvals);
        assert_eq!(adj[2], 1.0); // 0.8 * 3 = 2.4 => clamped to 1.0
    }
}
