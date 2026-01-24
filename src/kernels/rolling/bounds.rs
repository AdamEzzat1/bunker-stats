//! Window bounds calculation for different alignment strategies.

use super::config::Alignment;

/// Window bounds for a given position.
///
/// Bounds are [start, end) (half-open interval).
/// - start: inclusive first index
/// - end: exclusive last index
/// - actual_window: end - start (may be < config.window near edges)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowBounds {
    pub start: usize,
    pub end: usize,
}

impl WindowBounds {
    #[inline]
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        Self { start, end }
    }
    
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Trait for computing window bounds at each output position.
pub trait WindowBoundsCalculator {
    /// Compute bounds for output position k.
    ///
    /// # Arguments
    /// - k: output position (0-indexed)
    /// - n: input length
    /// - window: configured window size
    ///
    /// # Returns
    /// WindowBounds for this position (may be truncated near edges).
    fn bounds(&self, k: usize, n: usize, window: usize) -> WindowBounds;
}

/// Trailing window bounds (legacy behavior).
///
/// Output length: n - window + 1
/// Position k corresponds to input[k + window - 1].
/// Window at k: [k, k + window)
pub struct TrailingBounds;

impl WindowBoundsCalculator for TrailingBounds {
    #[inline]
    fn bounds(&self, k: usize, n: usize, window: usize) -> WindowBounds {
        debug_assert!(window > 0 && window <= n);
        debug_assert!(k < n.saturating_sub(window) + 1);
        
        WindowBounds::new(k, k + window)
    }
}

/// Centered window bounds.
///
/// Output length: n (same as input, "pandas-like")
/// Position k corresponds to input[k].
/// Window at k: [max(0, k - w/2), min(n, k + w/2 + 1))
/// where w = window size.
///
/// Near edges, window is truncated but centered as much as possible.
pub struct CenteredBounds;

impl WindowBoundsCalculator for CenteredBounds {
    #[inline]
    fn bounds(&self, k: usize, n: usize, window: usize) -> WindowBounds {
        debug_assert!(window > 0);
        debug_assert!(k < n);
        
        let half = window / 2;
        let start = k.saturating_sub(half);
        let end = (k + half + 1).min(n);
        
        WindowBounds::new(start, end)
    }
}

/// Get bounds calculator for alignment.
pub fn get_bounds_calculator(alignment: Alignment) -> Box<dyn WindowBoundsCalculator> {
    match alignment {
        Alignment::Trailing => Box::new(TrailingBounds),
        Alignment::Centered => Box::new(CenteredBounds),
    }
}

/// Compute output length for given alignment.
#[inline]
pub fn output_length(n: usize, window: usize, alignment: Alignment) -> usize {
    match alignment {
        Alignment::Trailing => {
            if window > n || n == 0 {
                0
            } else {
                n - window + 1
            }
        }
        Alignment::Centered => n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trailing_bounds() {
        let calc = TrailingBounds;
        
        // n=10, window=3
        // Output length = 10 - 3 + 1 = 8
        assert_eq!(calc.bounds(0, 10, 3), WindowBounds::new(0, 3));
        assert_eq!(calc.bounds(1, 10, 3), WindowBounds::new(1, 4));
        assert_eq!(calc.bounds(7, 10, 3), WindowBounds::new(7, 10));
    }
    
    #[test]
    fn test_centered_bounds() {
        let calc = CenteredBounds;
        
        // n=10, window=5
        // half = 2
        
        // k=0: [0, 3) (truncated on left)
        assert_eq!(calc.bounds(0, 10, 5), WindowBounds::new(0, 3));
        
        // k=2: [0, 5) (full window)
        assert_eq!(calc.bounds(2, 10, 5), WindowBounds::new(0, 5));
        
        // k=5: [3, 8) (full window)
        assert_eq!(calc.bounds(5, 10, 5), WindowBounds::new(3, 8));
        
        // k=9: [7, 10) (truncated on right)
        assert_eq!(calc.bounds(9, 10, 5), WindowBounds::new(7, 10));
    }
    
    #[test]
    fn test_output_length() {
        // Trailing
        assert_eq!(output_length(10, 3, Alignment::Trailing), 8);
        assert_eq!(output_length(5, 5, Alignment::Trailing), 1);
        
        // Centered
        assert_eq!(output_length(10, 3, Alignment::Centered), 10);
        assert_eq!(output_length(5, 5, Alignment::Centered), 5);
        
        // Edge cases: trailing
        assert_eq!(output_length(0, 3, Alignment::Trailing), 0);  // Empty array
        assert_eq!(output_length(3, 5, Alignment::Trailing), 0);  // Window too large
        
        // Edge cases: centered (always returns n)
        assert_eq!(output_length(0, 3, Alignment::Centered), 0);  // Empty array
        assert_eq!(output_length(3, 5, Alignment::Centered), 3);  // Window too large (still returns n)
    }
    
    #[test]
    fn test_bounds_len() {
        let b = WindowBounds::new(2, 5);
        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
        
        let empty = WindowBounds::new(5, 5);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}