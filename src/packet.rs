// Bits go: [..., a2, a1, a0]
// Where coefficient is a0 + 2 a1 + 4 a2 + ...
pub(crate) struct Packet<const LOG2: usize>(pub(crate) [u64; LOG2]);

macro_rules! repeat {
    ($times:ident, $me:ident[$indexing:ident => $idx:expr], $input_val:pat, ($($other:ident : $other_val:pat),*), $op:block) => {{
        let mut $indexing = 0;
        while $indexing < $times {
            let $input_val = $me.0[$idx];
            $(
                let $other_val = $other.0[$idx];
            )*

            let result = $op;

            $me.0[$idx] = result;

            $indexing += 1;
        }
        #[allow(path_statements)]
        { $me }
    }};

    ($times:ident, $me:ident, $input_val:pat, $op:block) => {
        repeat!($times, $me[x => x], $input_val, (), $op)
    };

    ($times:ident, $me:ident[$indexing:ident => $idx:expr], $input_val:pat, $op:block) => {
        repeat!($times, $me[$indexing => $idx], $input_val, (), $op)
    };

    ($times:ident, $me:ident, $input_val:pat, ($($other:ident : $other_val:pat),+), $op:block) => {
        repeat!($times, $me[x => x], $input_val, ($($other : $other_val),+), $op)
    }
}

macro_rules! reduce {
    ($times:ident, $acc:ident, $me:ident, $input_val:pat, ($($other:ident : $other_val:ident),*), $def:expr, $op:block) => {{
        let mut i = 0;
        let mut $acc = $def;

        while i < $times {
            let $input_val = $me.0[i];
            $(
                let $other_val = $other.0[i];
            )*

            $acc = $op;

            i += 1;
        }
        $acc
    }};

    ($times:ident, $acc:ident, $me:ident, $input_val:pat, $def:expr, $op:block) => {
        reduce!($times, $acc, $me, $input_val, (), $def, $op)
    }
}

impl<const LOG2: usize> Copy for Packet<LOG2> {}
impl<const LOG2: usize> Clone for Packet<LOG2> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const LOG2: usize> Packet<LOG2> {
    pub const fn new() -> Self {
        Self([0; LOG2])
    }

    pub const fn splat(mut val: u64) -> Self {
        let mut me = Self::new();

        repeat! {
            LOG2,
            me[x => LOG2 - x - 1],
            _,
            {
                let k = !((val & 1).wrapping_sub(1));

                val >>= 1;

                k
            }
        }
    }

    pub const fn from_int(mut val: u64) -> Self {
        let mut me = Self::new();
        repeat! {
            LOG2,
            me[x => LOG2 - x - 1],
            _,
            {
                let k = val & 1;

                val >>= 1;

                k
            }
        }
    }

    pub const fn lsh(mut self, by: usize) -> Self {
        repeat! {LOG2, self, x, { x << by }}
    }

    pub const fn rsh(mut self, by: usize) -> Self {
        repeat! {LOG2, self, x, { x >> by }}
    }

    pub const fn xor(mut self, other: Self) -> Self {
        repeat! {LOG2, self, x, (other: y), { x ^ y }}
    }

    pub const fn and(mut self, other: Self) -> Self {
        repeat! {LOG2, self, x, (other: y), { x & y }}
    }

    pub const fn or(mut self, other: Self) -> Self {
        repeat! {LOG2, self, x, (other: y), { x | y }}
    }

    pub const fn xor_u64(mut self, other: u64) -> Self {
        repeat! {LOG2, self, x, { x ^ other }}
    }

    pub const fn and_u64(mut self, other: u64) -> Self {
        repeat! {LOG2, self, x, { x & other }}
    }
    pub const fn or_u64(mut self, other: u64) -> Self {
        repeat! {LOG2, self, x, { x | other }}
    }

    pub const fn xor_reduce(self) -> u64 {
        reduce! {LOG2, acc, self, x, 0, { acc ^ x }}
    }

    pub const fn and_reduce(self) -> u64 {
        reduce! {LOG2, acc, self, x, 0, { acc & x }}
    }

    pub const fn or_reduce(self) -> u64 {
        reduce! {LOG2, acc, self, x, 0, { acc | x }}
    }

    pub const fn is_zero(self) -> bool {
        self.or_reduce() == 0
    }

    pub const fn not(mut self) -> Self {
        repeat! {LOG2, self, x, { !x }}
    }

    pub const fn right_shift_horizontal(mut self) -> (Self, u64) {
        let last = self.0[LOG2 - 1];

        let mut done = LOG2 - 1;

        while done > 0 {
            self.0[done] = self.0[done - 1];

            done -= 1;
        }

        self.0[0] = 0;

        (self, last)
    }

    pub const fn left_shift_horizontal(mut self) -> (u64, Self) {
        let first = self.0[0];

        let mut done = 0;

        while done < LOG2 - 1 {
            self.0[done] = self.0[done + 1];

            done += 1;
        }

        self.0[LOG2 - 1] = 0;

        (first, self)
    }

    pub const fn leading_zeros(self) -> u64 {
        self.or_reduce().leading_zeros() as u64
    }

    pub const fn extract_coefficient(mut self, idx: usize) -> u64 {
        let mask = 1u64 << idx;

        self = self.and_u64(mask);

        let mut result = 0;

        let mut i = 0;

        while i < LOG2 {
            result <<= 1;
            result += (self.0[i] != 0) as u64;

            i += 1;
        }

        result
    }

    pub const fn set_coeff(mut self, idx: usize, coeff: u64) -> Self {
        let mask = 1u64 << idx;
        let unset_mask = !mask;

        self = self.and_u64(unset_mask);

        let num = Self::from_int(coeff);

        self = self.or(num.lsh(idx));

        self
    }

    pub const fn from_coeffs(coeffs: &[u64]) -> Self {
        let mut me = Self::new();

        let mut idx = 0;
        while idx < coeffs.len() {
            let mut val = coeffs[idx];

            me = repeat! {
                LOG2,
                me[x => LOG2 - x - 1],
                mut w,
                {
                    let k = val & 1;

                    val >>= 1;

                    w <<= 1;

                    w | k
                }
            };

            idx += 1;
        }

        me
    }
}

impl<const LOG2: usize> std::fmt::Display for Packet<LOG2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.0[..])
    }
}
