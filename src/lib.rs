use std::{
    fmt::{Debug, Display, Write},
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use packet::Packet;

mod packet;

pub trait PolySettings<const SIZE: usize, const LOG2: usize>: Sized {
    const DEGREE: usize;
    const MODULO: usize;

    const OVERFLOW: FinitePoly<Self, SIZE, LOG2>;
}

pub struct FinitePoly<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> {
    pub(crate) internal: [Packet<LOG2>; SIZE],
    pub(crate) _phantom: PhantomData<T>,
}

// impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> Eq
//     for FinitePoly<T, SIZE, LOG2>
// {
// }

// impl<T: PolySettings<SIZE, LOG2>, const SIZE: usize, const LOG2: usize> PartialEq
//     for FinitePoly<T, SIZE, LOG2>
// {
//     fn eq(&self, other: &Self) -> bool {
//         Self::eq(*self, *other)
//     }
// }

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

    pub const ONE: Self = Self::from_int(1);

    const fn splat(value: u64) -> Self {
        Self {
            internal: [Packet::splat(value); SIZE],
            _phantom: PhantomData,
        }
    }

    const fn from_int(value: u64) -> Self {
        let mut me = Self::ZERO;
        me.internal[0] = Packet::from_int(value);

        me
    }

    pub const fn remove_false_zeros(mut self) -> Self {
        let mut done = 0;

        while done <= SIZE {
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

        while done < SIZE {
            if self.internal[done].or_reduce() != 0 {
                return false;
            }

            done += 1;
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

            let (bumped, new_carry) = new_carry.left_shift_bump();

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

    pub const fn sub_within(n: Packet<LOG2>, m: Packet<LOG2>) -> Packet<LOG2> {
        let mut result = n;
        let mut carry = m;
        let mut underflow_carry = Packet::new();

        while !carry.is_zero() || !underflow_carry.is_zero() {
            let sub = result.xor(carry).xor(underflow_carry);

            let new_carry = result
                .not()
                .and(carry.or(underflow_carry))
                .or(carry.and(underflow_carry));

            let (bumped, new_carry) = new_carry.left_shift_bump();

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
        let mut extracted_overflow = self.internal[SIZE - 1] & Self::EXTRA_POWER_SELECT;

        self.internal[SIZE - 1] ^= extracted_overflow;

        extracted_overflow >>= Self::EXTRA_RSH_TO_ONE;

        let shifted = self.unchecked_mulx(1);

        let multiplied_overflow = T::OVERFLOW.mul_modulo(extracted_overflow);
        shifted.add(multiplied_overflow)
    }

    pub const fn unchecked_mulx(self, num: usize) -> Self {
        if num == 0 {
            return self;
        }

        let mut result = [0; SIZE];

        let mut last_select = 0;
        let mut done = 0;
        let to_shift = Self::LAST_RSH_TO_ONE - (num - 1) * Self::LOG2;

        while done < num {
            last_select >>= Self::LOG2;
            last_select |= Self::LAST_SELECT;

            done += 1;
        }

        let mut i = 0;
        let mut carry = 0;

        while i < SIZE {
            let new_carry = (last_select & self.internal[i]) >> to_shift;
            result[i] = ((self.internal[i] << (Self::LOG2 * num)) & Self::CLIP_UNUSED_BITS) | carry;
            carry = new_carry;
            i += 1;
        }

        Self {
            internal: result,
            _phantom: PhantomData,
        }
    }

    pub const fn get_nth_coefficient(self, coeff: usize) -> u64 {
        if coeff > T::DEGREE {
            return 0;
        }

        let idx = coeff / Self::POWERS_PER_U64;
        let in_idx = coeff % Self::POWERS_PER_U64;

        let containing = self.internal[idx];
        let shifted = containing >> in_idx * Self::LOG2;
        shifted & Self::SELECT_FIRST
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

    const fn invert_in_modulo(n: u64) -> Option<u64> {
        let mut t = 0i64;
        let mut r = T::MODULO as u64;
        let mut new_t = 1i64;
        let mut new_r = n;

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient as i64 * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return None;
        }

        if t < 0 {
            t += T::MODULO as i64;
        }

        Some(t as _)
    }

    /// Returns: (gcd, n/gcd, m/gcd).
    const fn extended_euclidean_algo(n: u64, m: u64) -> (u64, u64, u64) {
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

    /// Returns (quotient, subtracted, remainder)
    fn subtract_largest_multiple(self, mut other: Self) -> Option<(Self, Self, Self)> {
        let my_degree = self.degree();
        let other_degree = other.degree();

        if my_degree < other_degree {
            return None;
        }

        let my_coeff = self.get_nth_coefficient(my_degree);
        let other_coeff = other.get_nth_coefficient(other_degree);

        let (_, mine_red, other_red) = Self::extended_euclidean_algo(my_coeff, other_coeff);

        let Some(inverse) = Self::invert_in_modulo(other_red) else {
            return None;
        };

        let multiplier = mine_red * inverse;
        let mut quotient = Self::integer_to_poly(multiplier);

        if my_degree > other_degree {
            other = other.unchecked_mulx(my_degree - other_degree);
            quotient = quotient.unchecked_mulx(my_degree - other_degree);
        }

        let multiplied = other.mul_modulo(multiplier);
        let diff = self.sub(multiplied);

        debug_assert!(!(diff.degree() == my_degree && !diff.is_zero()));
        // if diff.degree() == my_degree && !diff.is_zero() {
        //     panic!("Oh noes! {self}, {other}, {quotient}, {multiplied}, {diff}");
        // }

        Some((quotient, multiplied, diff))
    }

    pub fn invert(self) -> Option<Self> {
        let mut t = Self::ZERO;
        let mut r = T::OVERFLOW;
        let mut new_t = Self::ONE;
        let mut new_r = self;

        // Checking degree is faster than checking zero.
        while new_r.degree() > 0 || !new_r.is_zero() {
            let Some((quotient, _, remainder)) = Self::subtract_largest_multiple(r, new_r) else {
                println!("Invert None 0");
                return None;
            };

            (r, new_r) = (new_r, remainder);
            (t, new_t) = (new_t, t.sub(quotient.mul(new_t)));
        }

        if r.degree() > 0 {
            println!("Invert None 1: ({t}) -- ({r}) -- ({new_t}) -- ({new_r})");
            return None;
        }

        let r_as_integer = r.get_nth_coefficient(0);
        let Some(inverse) = Self::invert_in_modulo(r_as_integer) else {
            println!("Invert None 2");
            return None;
        };

        Some(t.mul_modulo(inverse))
    }

    pub const fn unchecked_make_from_coeffs_desc(coeffs: &[u64]) -> Self {
        let mut acc = Self::ZERO;

        let mut i = 0;

        while i != coeffs.len() {
            acc = acc.unchecked_mulx(1).add(Self::integer_to_poly(coeffs[i]));

            i += 1;
        }

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

    pub const fn iter() -> FinitePolyIterator<T, SIZE> {
        FinitePolyIterator {
            state: 0,
            _item: PhantomData,
        }
    }
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Debug for FinitePoly<T, SIZE> {
    fn fmt(&self, mut f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_full(&mut f)?;
        write!(f, " [")?;
        for val in self.internal[1..].iter().rev() {
            write!(f, "0b{val:b}, ")?;
        }
        write!(f, "0b{:b}", self.internal[0])?;
        write!(f, "]")
    }
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Display for FinitePoly<T, SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_filtered(f)
    }
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Mul<Self> for FinitePoly<T, SIZE> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Add<Self> for FinitePoly<T, SIZE> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Sub<Self> for FinitePoly<T, SIZE> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

pub struct FinitePolyIterator<T: PolySettings<SIZE>, const SIZE: usize> {
    state: u128,
    _item: PhantomData<FinitePoly<T, SIZE>>,
}

impl<T: PolySettings<SIZE>, const SIZE: usize> Iterator for FinitePolyIterator<T, SIZE> {
    type Item = FinitePoly<T, SIZE>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state == FinitePoly::<T, SIZE>::LAST_ITERATOR_ELEM {
            return None;
        }

        let mut acc = FinitePoly::<T, SIZE>::ZERO;

        for power in (0..T::DEGREE).rev() {
            let coeff = self.state / (T::MODULO as u128).pow(power as u32);
            let coeff = coeff % T::MODULO as u128;

            acc = acc
                .unchecked_mulx(1)
                .add(FinitePoly::<T, SIZE>::ONE.mul_modulo(coeff as u64));
        }

        self.state += 1;

        Some(acc)
    }
}

pub const fn log2(x: usize) -> usize {
    (usize::BITS - (x - 1).leading_zeros()) as _
}

const fn usable_bits(logarithm: usize) -> usize {
    let available_per = 64 - (64 % logarithm);

    available_per
}

const fn poly_num_u64s(logarithm: usize, deg: usize) -> usize {
    let total = logarithm * (deg + 1);
    let usable = usable_bits(logarithm);
    total.div_ceil(usable)
}

pub const fn get_size<T: PolySettings<0>>() -> usize {
    poly_num_u64s(log2(T::MODULO), T::DEGREE)
}

#[allow(unused_macros)]
macro_rules! make_ring {
    ($name:ident = $modulo:literal, $degree:literal, [$($coefficients:literal),+]) => {
        struct Settings;

        impl<const SIZE: usize> $crate::PolySettings<SIZE> for Settings {
            const DEGREE: usize = $degree;
            const MODULO: usize = $modulo;

            const OVERFLOW: $crate::FinitePoly<Self, SIZE> =
                <$crate::FinitePoly<Self, SIZE>>::unchecked_make_from_coeffs_desc(&[$($coefficients),+]);
        }

        type $name = $crate::FinitePoly<Settings, {$crate::get_size::<Settings>()}>;
    }
}

#[cfg(test)]
mod tests {
    make_ring! {
        F25 = 5, 5, [1, 4]
    }

    #[test]
    fn integer_to_poly() {
        let one_const = F25::ONE;
        let one_phi = F25::integer_to_poly(1);

        assert_eq!(one_const, one_phi);

        for i in 0..20 {
            let pre_reduced = i % 5;

            let value_1 = F25::integer_to_poly(i);
            let value_2 = F25::integer_to_poly(pre_reduced);
            let mut value_3 = F25::ZERO;

            for _ in 0..i {
                value_3 = value_3 + F25::ONE;
            }

            let mut value_4 = F25::ZERO;

            for _ in 0..pre_reduced {
                value_4 = value_4 + F25::ONE;
            }

            assert_eq!(value_1, value_2);
            assert_eq!(value_2, value_3);
            assert_eq!(value_3, value_4);
        }
    }

    #[test]
    fn equality() {
        for lhs in F25::iter() {
            for rhs in F25::iter() {
                let mut equal = true;
                for power in 0..5 {
                    let coeff_left = lhs.get_nth_coefficient(power) % 5;
                    let coeff_right = rhs.get_nth_coefficient(power) % 5;

                    equal &= coeff_left == coeff_right;
                }

                assert_eq!(equal, lhs == rhs);
            }
        }
    }
}
