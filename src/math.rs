use std::ops::{Add, Mul, Sub};

use num::Float;

/// Similar to numpy.interp
pub(crate) fn interp<F>(x_input: &[F], coords: impl IntoIterator<Item = (F, F)> + Clone) -> Vec<F>
where
    F: Add + Float + Mul + Sub,
{
    x_input
        .iter()
        .copied()
        .map(|x| {
            // TODO: Benchmark; prob faster to pull out into vec
            let mut iter = coords.clone().into_iter().enumerate().peekable();

            while let Some((i, (x1, y1))) = iter.next() {
                // Base cases
                if i == 0 && x < x1 {
                    return y1;
                }
                if iter.peek().is_none() && x > x1 {
                    return y1;
                }

                let Some(&(_, (x2, y2))) = iter.peek() else {
                    continue;
                };

                // Actual interpolation
                if x1 <= x && x <= x2 {
                    let y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);

                    return y;
                }
            }

            unreachable!()
        })
        .collect()
}

#[test]
fn test_interp() {
    let x = [0., 1., 1.5, 2.72, 3.24];
    let xs = [1., 2., 3.];
    let ys = [3., 2., 0.];

    assert_eq!(
        interp(&x, xs.into_iter().zip(ys.into_iter())),
        vec![3., 3., 2.5, 0.5599999999999996, 0.]
    );

    let x = [2.5, -1., 7.5];
    let xs = [0., 1., 2., 3., 4.5];
    let ys = [0., 2., 5., 3., 2.];

    assert_eq!(
        interp(&x, xs.into_iter().zip(ys.into_iter())),
        vec![4., 0., 2.]
    );
}
