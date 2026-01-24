//! Bitflags for selecting which statistics to compute.

use bitflags::bitflags;

bitflags! {
    /// Mask indicating which statistics to compute in fused kernel.
    ///
    /// Example:
    /// ```ignore
    /// let mask = StatsMask::MEAN | StatsMask::STD;
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StatsMask: u32 {
        const MEAN  = 1 << 0;
        const STD   = 1 << 1;
        const VAR   = 1 << 2;
        const COUNT = 1 << 3;
        const MIN   = 1 << 4;
        const MAX   = 1 << 5;
        
        // Common combinations
        const MEAN_STD = Self::MEAN.bits() | Self::STD.bits();
        const MEAN_VAR = Self::MEAN.bits() | Self::VAR.bits();
        const ALL = Self::MEAN.bits() 
                  | Self::STD.bits() 
                  | Self::VAR.bits()
                  | Self::COUNT.bits()
                  | Self::MIN.bits()
                  | Self::MAX.bits();
    }
}

impl StatsMask {
    /// Check if mean is requested.
    #[inline]
    pub fn has_mean(self) -> bool {
        self.contains(Self::MEAN)
    }
    
    /// Check if std is requested.
    #[inline]
    pub fn has_std(self) -> bool {
        self.contains(Self::STD)
    }
    
    /// Check if var is requested.
    #[inline]
    pub fn has_var(self) -> bool {
        self.contains(Self::VAR)
    }
    
    /// Check if count is requested.
    #[inline]
    pub fn has_count(self) -> bool {
        self.contains(Self::COUNT)
    }
    
    /// Check if min is requested.
    #[inline]
    pub fn has_min(self) -> bool {
        self.contains(Self::MIN)
    }
    
    /// Check if max is requested.
    #[inline]
    pub fn has_max(self) -> bool {
        self.contains(Self::MAX)
    }
    
    /// Parse from string slice (comma-separated).
    ///
    /// Example: "mean,std,var" -> MEAN | STD | VAR
    pub fn from_str_list(s: &str) -> Result<Self, String> {
        let mut mask = StatsMask::empty();
        
        for part in s.split(',').map(|p| p.trim().to_lowercase()) {
            match part.as_str() {
                "mean" => mask |= Self::MEAN,
                "std" => mask |= Self::STD,
                "var" => mask |= Self::VAR,
                "count" => mask |= Self::COUNT,
                "min" => mask |= Self::MIN,
                "max" => mask |= Self::MAX,
                "" => continue,
                _ => return Err(format!("Unknown stat: {}", part)),
            }
        }
        
        if mask.is_empty() {
            return Err("No valid statistics specified".to_string());
        }
        
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mask_combinations() {
        let m1 = StatsMask::MEAN | StatsMask::STD;
        assert!(m1.has_mean());
        assert!(m1.has_std());
        assert!(!m1.has_var());
        
        let m2 = StatsMask::MEAN_STD;
        assert_eq!(m1, m2);
    }
    
    #[test]
    fn test_from_str_list() {
        let m = StatsMask::from_str_list("mean,std").unwrap();
        assert!(m.has_mean());
        assert!(m.has_std());
        assert!(!m.has_var());
        
        let m2 = StatsMask::from_str_list("mean, std, var").unwrap();
        assert!(m2.has_mean());
        assert!(m2.has_std());
        assert!(m2.has_var());
        
        assert!(StatsMask::from_str_list("invalid").is_err());
        assert!(StatsMask::from_str_list("").is_err());
    }
    
    #[test]
    fn test_all_mask() {
        let all = StatsMask::ALL;
        assert!(all.has_mean());
        assert!(all.has_std());
        assert!(all.has_var());
        assert!(all.has_count());
        assert!(all.has_min());
        assert!(all.has_max());
    }
}
