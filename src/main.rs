use std::{hint::black_box, time::Instant};

use finite::{FinitePoly, PolySettings};

struct F4Settings;

impl<const SIZE: usize, const LOG2: usize> PolySettings<SIZE, LOG2> for F4Settings {
    // const DEGREE: usize = 2;
    const DEGREE: usize = 10;
    const MODULO: u64 = 5;
    // const MODULO: u64 = 3;

    // const OVERFLOW: FinitePoly<Self, SIZE, LOG2> =
    //     <FinitePoly<Self, SIZE, LOG2>>::make_from_coeffs_desc(&[4, 3]);
    const OVERFLOW: FinitePoly<Self, SIZE, LOG2> =
        <FinitePoly<Self, SIZE, LOG2>>::make_from_coeffs_desc(&[1, 1]);
    // const OVERFLOW: FinitePoly<Self, SIZE, LOG2> = <FinitePoly<Self, SIZE, LOG2>>::ZERO;
}

const SIZE: usize = finite::get_size::<F4Settings>();
const LOG2: usize = finite::get_log2::<F4Settings>();

const OVERFLOW: F4 = <F4Settings as PolySettings<SIZE, LOG2>>::OVERFLOW;

type F4 = FinitePoly<F4Settings, SIZE, LOG2>;

fn main() {
    let zero = F4::ZERO;
    let one = F4::ONE;

    // println!(
    //     "0 = {}, 1 = {:?}, val = {:?}",
    //     zero,
    //     one,
    //     F4::make_from_coeffs_desc(&[1])
    // );

    // let elems = F4::iter().collect::<Vec<_>>();

    let coeffs = [2, 4, 6, 7, 3, 0, 2, 3, 4];

    let start = Instant::now();

    // for x in &elems {
    //     for y in &elems {
    //         for z in &elems {
    //             black_box(x == y);
    //             black_box(y == z);
    //             black_box(x == z);
    //         }
    //     }
    // }
    // for x in F4::iter() {
    //     for y in F4::iter() {
    //         for z in F4::iter() {
    //             if x == y && y == z {
    //                 assert_eq!(x, z);
    //             }
    //         }
    //     }
    // }

    for _ in 0..100_000_000 {
        black_box(F4::make_from_coeffs_desc(&black_box(coeffs)));
    }

    let elapsed = start.elapsed();

    println!("Elapsed: {elapsed:?}");
    // for i in 0..10 {
    //     print!("{:?}", F4::from_int(i));

    //     let mut result = F4::ZERO;

    //     for _ in 0..i {
    //         result = result + one;
    //     }

    //     println!(" -- {:?}", result)
    // }

    // print!("        ");
    // for x in F4::iter() {
    //     print!("{:6}  ", format!("{}", x));
    // }
    // println!();

    // for x in F4::iter() {
    //     print!("{:6}  ", format!("{}", x));

    //     for y in F4::iter() {
    //         print!("{:6}  ", x == y);
    //     }

    //     println!();
    // }
    // let x = F4::ONE.unchecked_mulx(1);

    // println!("0     = {:?}", zero);
    // println!("1     = {:?}", one);
    // println!("1 + 1 = {:?}", one.add(one));
    // println!("4     = {:?}", one.add(one).add(one).add(one));
    // println!("x     = {:?}", x);
    // println!("x + x = {:?}", x.add(x));
    // println!("x + 1 = {:?}", x.add(one));
    // println!("x * x = {:?}", x.mul(x));
    // let seven = one.add(one).add(one).add(one).add(one).add(one).add(one);
    // println!("7     = {:?}", seven);
    // println!("7 + 7 = {:?}", seven.add(seven));
    // println!(
    //     "({:?}) * ({:?}) = {:?}",
    //     OVERFLOW,
    //     OVERFLOW,
    //     OVERFLOW * OVERFLOW
    // );
    // println!("spread overflow = 0b{:064b}", F4::SPREAD_MODULO_OVERFLOW);

    // println!("Log2: {}", finite::log2(3));

    // let mut acc = one;
    // for i in 0..20 {
    //     println!("x^{i} = {}", acc);
    //     acc = acc.mul_x();
    // }

    // println!("Select: {:b}", F4::EXTRA_POWER_SELECT);

    // println!("{} == {} : {}", zero, zero, zero == zero);
    // println!("{}", zero.sub(zero));

    // let iter = F4::iter();
    // for item in iter {
    //     println!("{}", item);
    // }
    // let mut count = 0;
    // println!("{:?} - {:?} = {:?}", zero, one, zero - one);
    // let three = F4::integer_to_poly(3);
    // let three_x = three.mul_x();
    // let one_plus_x = one.add(x);
    // let prod = three_x.mul(one_plus_x);
    // println!("{three:?} -- {three_x:?},\n{one:?} -- {x:?} -- {one_plus_x:?},\n{prod:?}");
    // println!("x^2 = {}", x.mul(x));
    // println!(
    //     "Expanded: {}",
    //     x.mul(x).mul_modulo(3).add(F4::integer_to_poly(3))
    // );

    // 'a: for lhs in F4::iter() {
    //     let inverse = lhs.invert();
    //     if let Some(y) = inverse {
    //         println!(
    //             "Finding inverse for {lhs}: {y} ({}) -- {} ({})",
    //             lhs * y,
    //             y * x.add(one),
    //             lhs * y * x.add(one)
    //         );
    //     }
    //     for rhs in F4::iter() {
    //         // println!("{lhs} * {rhs} = {}", lhs * rhs);
    //         let prod = lhs * rhs;
    //         if prod == F4::ONE {
    //             println!("{lhs} * {rhs} = {prod}!");

    //             continue 'a;
    //         }
    //     }

    //     println!("{lhs} inverse not found.");
    // println!("{lhs} has degree {}", lhs.degree());

    // count += 1;

    // if count > 10 {
    //     break;
    // }
    // }

    // println!(
    //     "{}, {}, {}, {}, {}",
    //     one,
    //     one.mul_x(),
    //     one.mul_x().mul_x(),
    //     one.mul_x().mul_x().mul_x(),
    //     one.mul_x().mul_x().mul_x().mul_x(),
    // );

    // let val = (one.unchecked_mulx(1) + one).unchecked_mulx(1);
    // println!("{val}");
}
