= `finitely` -- Optimized Finite Field Arithmetic

This crate implements arithmetic on rings of the form `(Z/nZ)[x]/(p(x))` with arbitrary `n` and `p`. It aims to be incredibly performant and feature-rich. 

== Example usage:
```
use finitely::make_ring;

make_ring! {
  pub Field25 = FieldSettings { Z % 5, x^2 = [2] };
}

let x = Field25::from_coeffs(&[1, 0]);

assert_eq!(x * x, 2);

// Notice that 3 * 2 = 6, which is 1 modulo 5.
// Therefore x * (3 * x) = x * x * 3 = 2 * 3 = 1.
// Therefore the inverse of x is 3x.
assert_eq!(x.invert(), Some(x * 3));

let x_plus_one = x + 1;
assert_eq!(x_plus_one.invert(), Some(x + 4));
assert_eq!(x_plus_one / x, x * 3 + 1);
```

== What _does_ this implement (but for non-mathematicians)?
This crate lets you represent polynomials that look like `a[0] + a[1]x + a[2]x^2 + ... + a[k]x^k`, with equivalences enforced to make it constant-size. Namely, each one of the coefficients `a[i]` is taken modulo `n` (the equivalence here is to say that `n` is equivalent to zero), and that the polynomial `p` is equivalent to zero. 

What does the second equivalence mean? Consider a polynomial that is of this shape: `p(x) = x^m + b[m-1]x^(m-1) + ... + b[2]x^2 + b[1]x + b[0]` (it is important that the coefficient of `x^m` be one). If we declare that `p(x)` is equivalent to zero, then we have essentially said that:
```
x^m = -(b[m-1]x^(m-1) + ... + b[2]x^2 + b[1]x + b[0])
```
So every polynomial with degree (highest power of `x`) which is greater than or equal to `m` can be rewritten as a polynomial of smaller degree. 

**Why do mathematicians care?**

If we pick `n` carefully (prime), and `p` carefully (irreducible), then the mathematical structure we get out is what is known as a field. A field is a mathematical structure where you have the four typical operations you are used to from `f32`: `+`, `-`, `*`, and `/`. If `n` and `p` are not chosen carefully, then `/` does not exist (but the other three still do). 

**How is this useful to a non-mathematician?**

This crate can be used to represent constant-length vectors of integers modulo `n`. If you do not use `*` or `/` (by another polynomial, and instead just regular integers), then you have an array of length `m` (where `m` is the degree of `p`) of integers modulo `n`.
