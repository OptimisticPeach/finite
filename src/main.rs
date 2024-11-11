use finite::{FinitePoly, PolySettings};

struct F4Settings;

impl<const SIZE: usize> PolySettings<SIZE> for F4Settings {
    const DEGREE: usize = 2;
    const MODULO: usize = 5;

    const OVERFLOW: FinitePoly<Self, SIZE> =
        <FinitePoly<Self, SIZE>>::unchecked_make_from_coeffs_desc(&[4, 3]);
}

const SIZE: usize = finite::get_size::<F4Settings>();

const OVERFLOW: F4 = <F4Settings as PolySettings<SIZE>>::OVERFLOW;

type F4 = FinitePoly<F4Settings, SIZE>;

fn main() {
    let zero = F4::ZERO;
    let one = F4::ONE;
    let x = F4::ONE.unchecked_mulx(1);

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

    'a: for lhs in F4::iter() {
        let inverse = lhs.invert();
        if let Some(y) = inverse {
            println!(
                "Finding inverse for {lhs}: {y} ({}) -- {} ({})",
                lhs * y,
                y * x.add(one),
                lhs * y * x.add(one)
            );
        }
        for rhs in F4::iter() {
            // println!("{lhs} * {rhs} = {}", lhs * rhs);
            let prod = lhs * rhs;
            if prod == F4::ONE {
                println!("{lhs} * {rhs} = {prod}!");

                continue 'a;
            }
        }

        println!("{lhs} inverse not found.");
        // println!("{lhs} has degree {}", lhs.degree());

        // count += 1;

        // if count > 10 {
        //     break;
        // }
    }

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
