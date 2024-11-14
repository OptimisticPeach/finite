use std::{
    fmt::{Debug, Display, Write},
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use packet::Packet;

mod packet;

pub trait PolySettings<const SIZE: usize, const LOG2: usize>: Sized {
    const DEGREE: usize;
    const MODULO: u64;

    const OVERFLOW: FinitePoly<Self, SIZE, LOG2>;
}

pub struct FinitePoly<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> {
    pub(crate) internal: [Packet<LOG2>; SIZE],
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Eq
    for FinitePoly<T, SIZE, LOG2>
{
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> PartialEq
    for FinitePoly<T, SIZE, LOG2>
{
    fn eq(&self, other: &Self) -> bool {
        Self::eq(*self, *other)
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Copy
    for FinitePoly<T, SIZE, LOG2>
{
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Clone
    for FinitePoly<T, SIZE, LOG2>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> FinitePoly<T, SIZE, LOG2> {
    pub const ZERO: Self = Self {
        internal: [Packet::<LOG2>::new(); SIZE],
        _phantom: PhantomData,
    };

    const FALSE_ZERO: Packet<LOG2> = Packet::splat(T::MODULO as u64 % (1u64 << LOG2));
    const OVERFLOW: Packet<LOG2> = Packet::splat((1u64 << LOG2) % T::MODULO as u64);
    const DEGREE_OVERFLOW_BIT: usize = T::DEGREE - 1 - Self::DEGREE_OVERFLOW_U64 * 64;
    const DEGREE_OVERFLOW_U64: usize = (T::DEGREE - 1) / 64;

    pub const ONE: Self = Self::from_int(1);

    pub const fn splat(value: u64) -> Self {
        Self {
            internal: [Packet::splat(value); SIZE],
            _phantom: PhantomData,
        }
    }

    pub const fn from_int(value: u64) -> Self {
        let mut me = Self::ZERO;
        me.internal[0] = Packet::from_int(value % T::MODULO);

        me
    }

    pub const fn remove_false_zeros(mut self) -> Self {
        let mut done = 0;

        while done < SIZE {
            let temp = self.internal[done];

            // xor detects any differences w/ false zero.
            // or reduction accumulates any differences into a single u64.
            // where there was a difference is now a 1, where there was
            // not (i.e. it is a false zero) is now a 0.
            let zeros_detect = temp.xor(Self::FALSE_ZERO).or_reduce();

            // and-ing will remove elements where zeros_detect is 0
            // which are places where we have a false zero.
            self.internal[done] = temp.and_u64(zeros_detect);

            done += 1;
        }

        self
    }

    pub const fn degree(mut self) -> usize {
        self = self.remove_false_zeros();

        let mut done = 1;

        while done <= SIZE {
            let to_detect = self.internal[SIZE - done];
            let leading = to_detect.leading_zeros();
            let first_one_idx = 64 - leading;

            if first_one_idx != 0 {
                let degree_total = first_one_idx + (SIZE - done) as u64 * 64;

                return degree_total as usize - 1;
            }

            done += 1;
        }

        0
    }

    pub const fn eq(self, other: Self) -> bool {
        let diff = self.sub(other);

        diff.is_zero()
    }

    pub const fn is_zero(mut self) -> bool {
        self = self.remove_false_zeros();
        let mut done = 0;

        while done < SIZE - 1 {
            if self.internal[done].or_reduce() != 0 {
                return false;
            }

            done += 1;
        }

        if self.internal[SIZE - 1].or_reduce() << (64 - T::DEGREE % 64) != 0 {
            return false;
        }

        true
    }

    const fn add_within(n: Packet<LOG2>, m: Packet<LOG2>) -> Packet<LOG2> {
        let mut result = n;
        let mut carry = m;
        let mut overflow_carry = Packet::new();

        while !carry.is_zero() || !overflow_carry.is_zero() {
            let add = result.xor(carry).xor(overflow_carry);
            // new_carry = (result & carry) | (result & overflow_carry) | (carry & overflow_carry).
            // That simplifies to this:
            let new_carry = result
                .and(carry.or(overflow_carry))
                .or(carry.and(overflow_carry));

            let (bumped, new_carry) = new_carry.left_shift_horizontal();

            let new_overflow = Self::OVERFLOW.and_u64(bumped);

            result = add;
            carry = new_carry;
            overflow_carry = new_overflow;
        }

        result
    }

    pub const fn add(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < SIZE {
            self.internal[i] = Self::add_within(self.internal[i], other.internal[i]);
            i += 1;
        }

        self
    }

    const fn sub_within(n: Packet<LOG2>, m: Packet<LOG2>) -> Packet<LOG2> {
        let mut result = n;
        let mut carry = m;
        let mut underflow_carry = Packet::new();

        while !carry.is_zero() || !underflow_carry.is_zero() {
            let sub = result.xor(carry).xor(underflow_carry);

            let new_carry = result
                .not()
                .and(carry.or(underflow_carry))
                .or(carry.and(underflow_carry));

            let (bumped, new_carry) = new_carry.left_shift_horizontal();

            let new_underflow = Self::OVERFLOW.and_u64(bumped);

            result = sub;
            carry = new_carry;
            underflow_carry = new_underflow;
        }

        result
    }

    pub const fn sub(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < SIZE {
            self.internal[i] = Self::sub_within(self.internal[i], other.internal[i]);
            i += 1;
        }

        self
    }

    pub const fn mul_modulo(self, by: u64) -> Self {
        let mut by = by % T::MODULO as u64;

        let mut acc = Self::ZERO;
        let mut power_2 = self;

        while by != 0 {
            if by & 1 == 1 {
                acc = acc.add(power_2);
            }

            by >>= 1;
            power_2 = power_2.add(power_2);
        }

        acc
    }

    pub const fn mul_x(mut self) -> Self {
        let extracted_overflow =
            self.internal[Self::DEGREE_OVERFLOW_U64].extract_coefficient(Self::DEGREE_OVERFLOW_BIT);

        let overflow = T::OVERFLOW.mul_modulo(extracted_overflow);

        self = self.unchecked_mulx(1);

        self.add(overflow)
    }

    pub const fn unchecked_mulx(mut self, power: usize) -> Self {
        let mut done = 0;

        let mut carry = Packet::new();

        while done != SIZE {
            let new_carry = self.internal[done].rsh(64 - power);
            self.internal[done] = self.internal[done].lsh(power).or(carry);

            carry = new_carry;
            done += 1;
        }

        self
    }

    pub const fn get_nth_coefficient(self, coeff: usize) -> u64 {
        if coeff > T::DEGREE {
            return 0;
        }

        let u64_idx = coeff / 64;
        let within_u64_idx = coeff % 64;

        self.internal[u64_idx].extract_coefficient(within_u64_idx)
    }

    pub fn mul(self, other: Self) -> Self {
        let mut acc = Self::ZERO;
        let mut power_x = self;

        let mut powers_done = 0;

        while powers_done < T::DEGREE {
            // println!("Acc: {acc}, Power X: {power_x}");
            let coeff = other.get_nth_coefficient(powers_done);

            if coeff != 0 {
                if coeff == 1 {
                    acc = acc.add(power_x);
                } else {
                    acc = acc.add(power_x.mul_modulo(coeff));
                }
            }

            power_x = power_x.mul_x();

            powers_done += 1;
        }

        acc
    }

    // const fn invert_in_modulo(n: u64) -> Option<u64> {
    //     let mut t = 0i64;
    //     let mut r = T::MODULO as u64;
    //     let mut new_t = 1i64;
    //     let mut new_r = n;

    //     while new_r != 0 {
    //         let quotient = r / new_r;
    //         (t, new_t) = (new_t, t - quotient as i64 * new_t);
    //         (r, new_r) = (new_r, r - quotient * new_r);
    //     }

    //     if r > 1 {
    //         return None;
    //     }

    //     if t < 0 {
    //         t += T::MODULO as i64;
    //     }

    //     Some(t as _)
    // }

    /// Returns: (gcd, n/gcd, m/gcd).
    // const fn extended_euclidean_algo(n: u64, m: u64) -> (u64, u64, u64) {
    //     let mut old_r = n as i64;
    //     let mut old_s = 1;
    //     let mut old_t = 0;
    //     let mut r = m as i64;
    //     let mut s = 0;
    //     let mut t = 1;

    //     while r != 0 {
    //         let q = old_r / r;
    //         (old_r, r) = (r, old_r - q * r);
    //         (old_s, s) = (s, old_s - q * s);
    //         (old_t, t) = (t, old_t - q * t);
    //     }

    //     (old_r.abs() as u64, t.abs() as u64, s.abs() as u64)
    // }

    // /// Returns (quotient, subtracted, remainder)
    // fn subtract_largest_multiple(self, mut other: Self) -> Option<(Self, Self, Self)> {
    //     let my_degree = self.degree();
    //     let other_degree = other.degree();

    //     if my_degree < other_degree {
    //         return None;
    //     }

    //     let my_coeff = self.get_nth_coefficient(my_degree);
    //     let other_coeff = other.get_nth_coefficient(other_degree);

    //     let (_, mine_red, other_red) = Self::extended_euclidean_algo(my_coeff, other_coeff);

    //     let Some(inverse) = Self::invert_in_modulo(other_red) else {
    //         return None;
    //     };

    //     let multiplier = mine_red * inverse;
    //     let mut quotient = Self::integer_to_poly(multiplier);

    //     if my_degree > other_degree {
    //         other = other.unchecked_mulx(my_degree - other_degree);
    //         quotient = quotient.unchecked_mulx(my_degree - other_degree);
    //     }

    //     let multiplied = other.mul_modulo(multiplier);
    //     let diff = self.sub(multiplied);

    //     debug_assert!(!(diff.degree() == my_degree && !diff.is_zero()));
    //     // if diff.degree() == my_degree && !diff.is_zero() {
    //     //     panic!("Oh noes! {self}, {other}, {quotient}, {multiplied}, {diff}");
    //     // }

    //     Some((quotient, multiplied, diff))
    // }

    // pub fn invert(self) -> Option<Self> {
    //     let mut t = Self::ZERO;
    //     let mut r = T::OVERFLOW;
    //     let mut new_t = Self::ONE;
    //     let mut new_r = self;

    //     // Checking degree is faster than checking zero.
    //     while new_r.degree() > 0 || !new_r.is_zero() {
    //         let Some((quotient, _, remainder)) = Self::subtract_largest_multiple(r, new_r) else {
    //             println!("Invert None 0");
    //             return None;
    //         };

    //         (r, new_r) = (new_r, remainder);
    //         (t, new_t) = (new_t, t.sub(quotient.mul(new_t)));
    //     }

    //     if r.degree() > 0 {
    //         println!("Invert None 1: ({t}) -- ({r}) -- ({new_t}) -- ({new_r})");
    //         return None;
    //     }

    //     let r_as_integer = r.get_nth_coefficient(0);
    //     let Some(inverse) = Self::invert_in_modulo(r_as_integer) else {
    //         println!("Invert None 2");
    //         return None;
    //     };

    //     Some(t.mul_modulo(inverse))
    // }

    pub const fn make_from_coeffs_desc(mut coeffs: &[u64]) -> Self {
        let to_do = if coeffs.len() > T::DEGREE {
            T::DEGREE
        } else {
            coeffs.len()
        };

        (_, coeffs) = coeffs.split_at(coeffs.len() - to_do);

        let last_block_length = coeffs.len() % 64;

        let (last_block, mut coeffs) = coeffs.split_at(last_block_length);

        let last_block = Packet::from_coeffs(last_block);

        let mut acc = Self::ZERO;

        let mut insertion_idx = 0;

        while coeffs.len() != 0 {
            let (rest, last) = coeffs.split_at(coeffs.len() - 64);

            coeffs = rest;

            acc.internal[insertion_idx] = Packet::from_coeffs(last);

            insertion_idx += 1;
        }

        acc.internal[insertion_idx] = last_block;

        acc
    }

    pub fn format_full(self, mut w: impl Write) -> std::fmt::Result {
        for i in (1..T::DEGREE).rev() {
            let coeff = self.get_nth_coefficient(i) % T::MODULO as u64;

            write!(w, "{coeff}x^{i} + ")?;
        }

        write!(w, "{}", self.get_nth_coefficient(0) % T::MODULO as u64)
    }

    pub fn format_filtered(self, mut w: impl Write) -> std::fmt::Result {
        if self == Self::ZERO {
            return write!(w, "0");
        }

        let mut seen_first = false;

        for i in (1..T::DEGREE).rev() {
            let coeff = self.get_nth_coefficient(i) % T::MODULO as u64;

            if coeff != 0 {
                if seen_first {
                    write!(w, " + ")?;
                } else {
                    seen_first = true;
                }

                if coeff != 1 {
                    write!(w, "{coeff}")?;
                }

                write!(w, "x")?;

                if i != 1 {
                    write!(w, "^{i}")?;
                }
            }
        }

        let zeroth = self.get_nth_coefficient(0) % T::MODULO as u64;

        if zeroth != 0 {
            if seen_first {
                write!(w, " + {zeroth}")?;
            } else {
                write!(w, "{zeroth}")?;
            }
        }

        Ok(())
    }

    pub fn iter() -> FinitePolyIterator<T, SIZE, LOG2> {
        FinitePolyIterator {
            coeffs: vec![0; T::DEGREE],
            _item: PhantomData,
        }
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Debug
    for FinitePoly<T, SIZE, LOG2>
{
    fn fmt(&self, mut f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_full(&mut f)?;
        write!(f, " [")?;
        for val in self.internal[1..].iter().rev() {
            write!(f, "{val}, ")?;
        }
        write!(f, "{}", self.internal[0])?;
        write!(f, "]")
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Display
    for FinitePoly<T, SIZE, LOG2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_filtered(f)
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Mul<Self>
    for FinitePoly<T, SIZE, LOG2>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Add<Self>
    for FinitePoly<T, SIZE, LOG2>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Sub<Self>
    for FinitePoly<T, SIZE, LOG2>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

pub struct FinitePolyIterator<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> {
    coeffs: Vec<u64>,
    _item: PhantomData<FinitePoly<T, SIZE, LOG2>>,
}

impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Iterator
    for FinitePolyIterator<T, SIZE, LOG2>
{
    type Item = FinitePoly<T, SIZE, LOG2>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.coeffs.len() == 0 {
            return None;
        }

        let val = Self::Item::make_from_coeffs_desc(&self.coeffs[..]);

        let mut carry = 1;
        let mut is_zero = true;

        for elem in self.coeffs.iter_mut().rev() {
            let mut new_val = *elem + carry;

            carry = new_val / T::MODULO as u64;
            new_val -= carry * T::MODULO as u64;

            *elem = new_val;

            is_zero &= new_val == 0;

            if carry == 0 {
                break;
            }
        }

        if is_zero {
            self.coeffs = vec![];
        }

        Some(val)
    }
}

pub const fn log2(x: u64) -> usize {
    (64 - (x - 1).leading_zeros()) as _
}

pub const fn get_size<T: PolySettings<0, 0>>() -> usize {
    T::DEGREE.div_ceil(64)
}

pub const fn get_log2<T: PolySettings<0, 0>>() -> usize {
    log2(T::MODULO)
}

#[allow(unused_macros)]
macro_rules! make_ring {
    ($name:ident = % $modulo:literal ^ $degree:literal, [$($coefficients:literal),+]) => {
        struct Settings;

        impl<const SIZE: usize, const LOG2: usize> $crate::PolySettings<SIZE, LOG2> for Settings {
            const DEGREE: usize = $degree;
            const MODULO: u64 = $modulo;

            const OVERFLOW: $crate::FinitePoly<Self, SIZE, LOG2> =
                <$crate::FinitePoly<Self, SIZE, LOG2>>::make_from_coeffs_desc(&[$($coefficients),+]);
        }

        type $name = $crate::FinitePoly<
            Settings,
            { $crate::get_size::<Settings>() },
            { $crate::log2($modulo) },
        >;
    };
}

#[cfg(test)]
mod tests {
    make_ring! {
        F125 = %5 ^3, [2, 2]
    }

    #[test]
    fn integer_to_poly() {
        let one_const = F125::ONE;
        let one_phi = F125::from_int(1);

        assert_eq!(one_const, one_phi);

        for i in 0..20 {
            let pre_reduced = i % 5;

            let value_1 = F125::from_int(i);
            let value_2 = F125::from_int(pre_reduced);
            let mut value_3 = F125::ZERO;

            for _ in 0..i {
                value_3 = value_3 + F125::ONE;
            }

            let mut value_4 = F125::ZERO;

            for _ in 0..pre_reduced {
                value_4 = value_4 + F125::ONE;
            }

            assert_eq!(value_1, value_2);
            assert_eq!(value_2, value_3);
            assert_eq!(value_3, value_4);
        }
    }

    #[test]
    fn coeff_equality() {
        for lhs in F125::iter() {
            let left = [0, 1, 2, 3, 4].map(|x| lhs.get_nth_coefficient(x));
            for rhs in F125::iter() {
                let mut equal = true;
                for power in 0..5 {
                    let coeff_left = left[power];
                    let coeff_right = rhs.get_nth_coefficient(power) % 5;

                    equal &= coeff_left == coeff_right;
                }

                assert_eq!(equal, lhs == rhs);
            }
        }
    }

    #[test]
    fn equality_is_equality() {
        // An equivalence relation ~ satisfies:
        // 1. x ~ x
        // 2. x ~ y ==> y ~ x
        // 3. x ~ y and y ~ z ==> x ~ z

        // Test identity.
        for x in F125::iter() {
            assert_eq!(x, x);
        }

        // Test reflexivity.
        for x in F125::iter() {
            for y in F125::iter() {
                assert_eq!(x == y, y == x);
            }
        }

        // Test transitivity.
        for x in F125::iter() {
            for y in F125::iter() {
                for z in F125::iter() {
                    if x == y && y == z {
                        assert_eq!(x, z);
                    }
                }
            }
        }
    }

    // We happen to have a field. The field axioms are:
    // 1.  Exists `+: F x F -> F` (done.)
    // 2.  Exists `*: F x F -> F` (done.)
    // 3.  For all: `x + y = y + x`.
    // 4.  For all: `x * y = y * x`.
    // 5.  For all: `x + (y + z) = (x + y) + z`
    // 6.  For all: `x * (y * z) = (x * y) * z`
    // 7.  Exists `0 in F`: For all: `x + 0 = x`.
    // 8.  Exists `1 in F`: For all: `x * 1 = x`.
    // 9.  For all: `x * (y + z) = x * y + x * z`.
    // 10. For all `x in F`: Exists `y in F`: `x + y = 0`
    // The previous 10 give us a Commutative Unital Ring
    // (from now on, this is just a ring). These two extra
    // axioms make it a field:
    // 11. `0 != 1`.
    // 12. For all `x in F`: `x != y` implies Exists `y in F`: `x * y = 1`

    #[test]
    fn addition_commutes() {
        for x in F125::iter() {
            for y in F125::iter() {
                assert_eq!(x + y, y + x);
            }
        }
    }

    #[test]
    fn multiplication_commutes() {
        for x in F125::iter() {
            for y in F125::iter() {
                assert_eq!(x * y, y * x);
            }
        }
    }

    #[test]
    fn addition_associates() {
        for x in F125::iter() {
            for y in F125::iter() {
                for z in F125::iter() {
                    assert_eq!(x + (y + z), (x + y) + z);
                }
            }
        }
    }

    #[test]
    fn multiplication_associates() {
        for x in F125::iter() {
            for y in F125::iter() {
                for z in F125::iter() {
                    assert_eq!(x * (y * z), (x * y) * z);
                }
            }
        }
    }

    #[test]
    fn zero_is_zero() {
        for x in F125::iter() {
            assert_eq!(x + F125::ZERO, x);
        }
    }

    #[test]
    fn one_is_one() {
        for x in F125::iter() {
            assert_eq!(x * F125::ONE, x);
        }
    }

    #[test]
    fn multiplication_distributes() {
        for x in F125::iter() {
            for y in F125::iter() {
                for z in F125::iter() {
                    assert_eq!(x * (y + z), (x * y) + (x * z));
                }
            }
        }
    }

    #[test]
    fn additive_inverses() {
        'a: for x in F125::iter() {
            for y in F125::iter() {
                if x + y == F125::ZERO {
                    continue 'a;
                }
            }

            panic!("Additive inverse for {x} not found!");
        }
    }

    #[test]
    fn zero_is_not_one() {
        assert_ne!(F125::ZERO, F125::ONE);
    }

    #[test]
    fn multiplicative_inverse() {
        'a: for x in F125::iter() {
            if x == F125::ZERO {
                continue;
            }

            for y in F125::iter() {
                if x * y == F125::ONE {
                    continue 'a;
                }
            }

            panic!("Multiplicative inverse for {x} not found!");
        }
    }
}
