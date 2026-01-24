//! Configuration types for rolling window operations.

use std::fmt;

/// Window alignment strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// Trailing window: window ends at current position.
    /// For position i with window w: uses [i-w+1, i] (inclusive).
    Trailing,
    
    /// Centered window: window is centered at current position.
    /// For position i with window w: uses [i-w/2, i+w/2] (inclusive).
    /// Actual window size may be smaller near edges.
    Centered,
}

impl Default for Alignment {
    fn default() -> Self {
        Self::Trailing
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trailing => write!(f, "trailing"),
            Self::Centered => write!(f, "centered"),
        }
    }
}

/// NaN handling policy for rolling windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanPolicy {
    /// Propagate: any NaN in window -> result is NaN.
    /// Strictest policy; matches legacy behavior.
    Propagate,
    
    /// Ignore: skip NaNs when computing statistics.
    /// If valid_count >= min_periods, compute stats on valid values.
    /// If valid_count < min_periods, result is NaN.
    Ignore,
    
    /// RequireMinPeriods: like Ignore, but explicitly enforce min_periods.
    /// This is the same as Ignore but makes the requirement explicit.
    /// (Functionally identical to Ignore; kept for API clarity.)
    RequireMinPeriods,
}

impl Default for NanPolicy {
    fn default() -> Self {
        Self::Propagate
    }
}

impl fmt::Display for NanPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Propagate => write!(f, "propagate"),
            Self::Ignore => write!(f, "ignore"),
            Self::RequireMinPeriods => write!(f, "require_min_periods"),
        }
    }
}

/// Configuration for rolling window operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RollingConfig {
    /// Window size (number of observations).
    pub window: usize,
    
    /// Minimum number of valid (non-NaN) observations required.
    /// If None, defaults to window size (all observations must be valid).
    /// Must satisfy: 1 <= min_periods <= window.
    pub min_periods: Option<usize>,
    
    /// Window alignment strategy.
    pub alignment: Alignment,
    
    /// NaN handling policy.
    pub nan_policy: NanPolicy,
}

impl RollingConfig {
    /// Create a new RollingConfig with validation.
    ///
    /// # Errors
    /// Returns error string if:
    /// - window < 1
    /// - min_periods > window
    /// - min_periods == 0
    pub fn new(
        window: usize,
        min_periods: Option<usize>,
        alignment: Alignment,
        nan_policy: NanPolicy,
    ) -> Result<Self, String> {
        // Validate window
        if window == 0 {
            return Err("window must be >= 1".to_string());
        }
        
        // Validate min_periods
        if let Some(mp) = min_periods {
            if mp == 0 {
                return Err("min_periods must be >= 1 (or None)".to_string());
            }
            if mp > window {
                return Err(format!(
                    "min_periods ({}) cannot exceed window ({})",
                    mp, window
                ));
            }
        }
        
        Ok(Self {
            window,
            min_periods,
            alignment,
            nan_policy,
        })
    }
    
    /// Create a trailing window config (legacy compatibility).
    pub fn trailing(window: usize) -> Self {
        Self {
            window,
            min_periods: None,
            alignment: Alignment::Trailing,
            nan_policy: NanPolicy::Propagate,
        }
    }
    
    /// Create a centered window config.
    pub fn centered(window: usize) -> Self {
        Self {
            window,
            min_periods: None,
            alignment: Alignment::Centered,
            nan_policy: NanPolicy::Propagate,
        }
    }
    
    /// Get effective min_periods (defaults to window if None).
    #[inline]
    pub fn effective_min_periods(&self) -> usize {
        self.min_periods.unwrap_or(self.window)
    }
    
    /// Check if this config matches legacy behavior.
    pub fn is_legacy_compatible(&self) -> bool {
        self.alignment == Alignment::Trailing
            && self.nan_policy == NanPolicy::Propagate
            && self.min_periods.is_none()
    }
}

impl Default for RollingConfig {
    fn default() -> Self {
        Self {
            window: 3,
            min_periods: None,
            alignment: Alignment::Trailing,
            nan_policy: NanPolicy::Propagate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_validation() {
        // Valid configs
        assert!(RollingConfig::new(5, None, Alignment::Trailing, NanPolicy::Propagate).is_ok());
        assert!(RollingConfig::new(5, Some(3), Alignment::Centered, NanPolicy::Ignore).is_ok());
        
        // Invalid: window = 0
        assert!(RollingConfig::new(0, None, Alignment::Trailing, NanPolicy::Propagate).is_err());
        
        // Invalid: min_periods > window
        assert!(RollingConfig::new(5, Some(10), Alignment::Trailing, NanPolicy::Propagate).is_err());
        
        // Invalid: min_periods = 0
        assert!(RollingConfig::new(5, Some(0), Alignment::Trailing, NanPolicy::Propagate).is_err());
    }
    
    #[test]
    fn test_effective_min_periods() {
        let cfg1 = RollingConfig::new(5, None, Alignment::Trailing, NanPolicy::Propagate).unwrap();
        assert_eq!(cfg1.effective_min_periods(), 5);
        
        let cfg2 = RollingConfig::new(5, Some(3), Alignment::Trailing, NanPolicy::Ignore).unwrap();
        assert_eq!(cfg2.effective_min_periods(), 3);
    }
    
    #[test]
    fn test_legacy_compatibility() {
        let cfg1 = RollingConfig::trailing(5);
        assert!(cfg1.is_legacy_compatible());
        
        let cfg2 = RollingConfig::new(5, Some(3), Alignment::Trailing, NanPolicy::Propagate).unwrap();
        assert!(!cfg2.is_legacy_compatible());
        
        let cfg3 = RollingConfig::centered(5);
        assert!(!cfg3.is_legacy_compatible());
    }
}
