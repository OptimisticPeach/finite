pub(crate) const fn invert_in_modulo(modulo: u64, n: u64) -> Option<u64> {
    let mut t = 0i64;
    let mut r = modulo as i64;
    let mut new_t = 1i64;
    let mut new_r = n as i64;

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient as i64 * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        return None;
    }

    if t < 0 {
        t += modulo as i64;
    }

    Some(t as _)
}

/// Returns: (gcd, n/gcd, m/gcd).
pub(crate) const fn extended_euclidean_algo(n: u64, m: u64) -> (u64, u64, u64) {
    let mut old_r = n as i64;
    let mut old_s = 1;
    let mut old_t = 0;
    let mut r = m as i64;
    let mut s = 0;
    let mut t = 1;

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
    }

    (old_r.abs() as u64, t.abs() as u64, s.abs() as u64)
}

/// Finds a number x such that mx = n (i.e. x = n/m).
pub(crate) const fn divide_modulo(modulo: u64, n: u64, m: u64) -> Option<u64> {
    let (_gcd, n_gcd, m_gcd) = extended_euclidean_algo(n, m);

    let Some(m_inverse) = invert_in_modulo(modulo, m_gcd) else {
        return None;
    };

    Some((m_inverse * n_gcd) % modulo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_inverse() {
        assert_eq!(invert_in_modulo(5, 4), Some(4));
        assert_eq!(invert_in_modulo(6, 5), Some(5));
        assert_eq!(invert_in_modulo(6, 3), None);
        assert_eq!(invert_in_modulo(12, 11), Some(11));
        assert_eq!(invert_in_modulo(12, 1), Some(1));
        assert_eq!(invert_in_modulo(12, 2), None);
    }

    #[test]
    fn gcd() {
        assert_eq!(extended_euclidean_algo(6, 10), (2, 3, 5));
        assert_eq!(extended_euclidean_algo(7, 5), (1, 7, 5));
    }

    #[test]
    fn divide() {
        assert_eq!(divide_modulo(35, 10, 15), Some(24));
        assert_eq!(divide_modulo(49, 14, 21), Some(17));

        assert_eq!(divide_modulo(35, 10, 14), None);
        assert_eq!(divide_modulo(35, 10, 21), None);
    }
}
