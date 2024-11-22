use std::{hint::black_box, time::Instant};

use finitely::make_ring;

// make_ring! {
//     F125 = %5 ^3, [2, 2]
// }

// 144444444
//

make_ring! {
    Nonsense = NonsenseSettings { Z % 7901, x^7 = [7900, 0, 7896, 0, 7884] };
    Nonsense2 = Nonsense2Settings { Z % 7901, x^7 = [7900, 0, 7896, 0, 7884] };
}

// make_ring! {
//     Other = Z % 17, x^2 = [1, 1]
// }

fn main() {
    // let value = Nonsense::ONE;

    // println!("{}", (value * value) == value);
    // let zero = F125::ZERO;
    // let one = F125::ONE;

    // for elem in [zero, one, <Settings>::OVERFLOW] {
    //     match elem.divide_quotient_poly_by_self() {
    //         Some((div, quot)) => println!("{div}\n{quot}"),
    //         None => println!("None\nNone"),
    //     }
    // }

    // println!(
    //     "{:?}\n{:?}\n{:?}",
    //     zero.divide_quotient_poly_by_self(),
    //     one.divide_quotient_poly_by_self(),
    //     <Settings as PolySettings<1, 1>>::OVERFLOW
    // );
    // for x in Nonsense::iter() {
    //     let computed_inverse = x.invert();

    //     let mut found_inverse = None;
    //     for y in Nonsense::iter() {
    //         if x * y == Nonsense::ONE {
    //             found_inverse = Some(y);
    //             break;
    //         }
    //     }

    //     assert_eq!(computed_inverse, found_inverse, "Poly: {x}");

    //     match computed_inverse {
    //         Some(i) => println!("({x})^-1 = {i}"),
    //         None => println!("Multiplicative inverse for {x} not found!"),
    //     }
    // }

    let val = Nonsense::from_coeffs(&[1, 2, 3, 4, 5]);
    let inverse = val.invert().unwrap();

    println!("{val}");
    println!("{:?}", inverse);
    println!("{}", val * inverse);

    let mut avg_time_ms = 0.0f64;

    for _ in 0..20 {
        let start = Instant::now();

        for _ in 0..100_000 {
            let inverse = black_box(val).invert().unwrap();
            black_box(inverse);
        }

        let elapsed = start.elapsed();

        println!("{elapsed:?}");
        avg_time_ms += elapsed.as_secs_f64() * 1000.0;
    }

    println!(
        "Average time to divide: {}ms",
        avg_time_ms / (20.0 * 100000.0)
    );

    let mut avg_time_ms = 0.0f64;

    for _ in 0..20 {
        let start = Instant::now();

        for _ in 0..100_000 {
            let inverse = black_box(inverse).invert().unwrap();
            black_box(inverse);
        }

        let elapsed = start.elapsed();

        println!("{elapsed:?}");
        avg_time_ms += elapsed.as_secs_f64() * 1000.0;
    }

    println!(
        "Average time to divide: {}ms",
        avg_time_ms / (20.0 * 100000.0)
    );

    // println!(
    //     "{val} has:\ninverse: {inverse}\nproduct is {}",
    //     val * inverse
    // );
}
