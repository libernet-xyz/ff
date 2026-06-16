/// Adds up two `u64`s with overflow and returns the (sum, carry) pair.
#[inline(always)]
pub const fn add(a: u64, b: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Subtracts `b` from `a` with wraparound and returns the (difference, borrow) pair.
#[inline(always)]
pub const fn sub(a: u64, b: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub(b as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Multiplies `a` and `b` with carry & overflow, returning the (product, carry) pair.
#[inline(always)]
pub const fn mul(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) * (b as u128) + carry as u128;
    (ret as u64, (ret >> 64) as u64)
}

/// Adds `a`, `b`, and `carry` and returns the (sum, carry) pair.
#[inline(always)]
pub const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Subtracts `b` and `borrow` from `a` and returns the (difference, borrow) pair.
#[inline(always)]
pub const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
    (ret as u64, (ret >> 64) as u64)
}

/// Computes `a + b * c + carry` and returns the (low, high) pair.
#[inline(always)]
pub const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + ((b as u128) * (c as u128)) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(0, 0), (0, 0));
        assert_eq!(add(1, 2), (3, 0));
        assert_eq!(add(u64::MAX - 1, 1), (u64::MAX, 0));
        assert_eq!(add(u64::MAX, 1), (0, 1));
        assert_eq!(add(u64::MAX, u64::MAX), (u64::MAX - 1, 1));
    }

    #[test]
    fn test_sub() {
        assert_eq!(sub(0, 0), (0, 0));
        assert_eq!(sub(5, 3), (2, 0));
        assert_eq!(sub(u64::MAX, u64::MAX), (0, 0));
        assert_eq!(sub(0, 1), (u64::MAX, u64::MAX));
        assert_eq!(sub(3, 5), (u64::MAX - 1, u64::MAX));
    }

    #[test]
    fn test_mul() {
        assert_eq!(mul(0, 0, 0), (0, 0));
        assert_eq!(mul(2, 3, 0), (6, 0));
        assert_eq!(mul(1, u64::MAX, 0), (u64::MAX, 0));
        assert_eq!(mul(2, 3, 1), (7, 0));
        assert_eq!(mul(0, 0, u64::MAX), (u64::MAX, 0));
        assert_eq!(mul(u64::MAX, u64::MAX, 0), (1, u64::MAX - 1));
        assert_eq!(mul(u64::MAX, u64::MAX, u64::MAX), (0, u64::MAX));
    }

    #[test]
    fn test_adc() {
        assert_eq!(adc(0, 0, 0), (0, 0));
        assert_eq!(adc(1, 2, 0), (3, 0));
        assert_eq!(adc(1, 2, 1), (4, 0));
        assert_eq!(adc(u64::MAX, 0, 1), (0, 1));
        assert_eq!(adc(u64::MAX, 1, 0), (0, 1));
        assert_eq!(adc(u64::MAX, u64::MAX, 1), (u64::MAX, 1));
    }

    #[test]
    fn test_sbb() {
        assert_eq!(sbb(5, 3, 0), (2, 0));
        assert_eq!(sbb(3, 3, 0), (0, 0));
        assert_eq!(sbb(u64::MAX, u64::MAX, 0), (0, 0));
        assert_eq!(sbb(5, 3, u64::MAX), (1, 0));
        assert_eq!(sbb(1, 0, u64::MAX), (0, 0));
        assert_eq!(sbb(0, 1, 0), (u64::MAX, u64::MAX));
        assert_eq!(sbb(0, 0, u64::MAX), (u64::MAX, u64::MAX));
        assert_eq!(sbb(0, u64::MAX, u64::MAX), (0, u64::MAX));
    }

    #[test]
    fn test_mac() {
        assert_eq!(mac(0, 0, 0, 0), (0, 0));
        assert_eq!(mac(1, 2, 3, 0), (7, 0));
        assert_eq!(mac(0, 1, u64::MAX, 0), (u64::MAX, 0));
        assert_eq!(mac(0, 2, 3, 1), (7, 0));
        assert_eq!(mac(u64::MAX, 1, 1, 0), (0, 1));
        assert_eq!(
            mac(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            (u64::MAX, u64::MAX)
        );
    }
}
