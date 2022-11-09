use std::iter::repeat;
use std::ops::{Add, Mul, Sub};

use num::Float;

// Could maybe use ndarray for this in the future?
pub struct Matrix<F> {
    values: Vec<F>,
    shape: (usize, usize),
}

impl<F> Matrix<F> {
    pub fn new(values: Vec<F>, shape: (usize, usize)) -> Self {
        debug_assert_eq!(values.len(), shape.0 * shape.1);

        Self { values, shape }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

impl<F: Float> Matrix<F> {
    pub fn tile(&self, shape: (usize, usize)) -> Self {
        repeat(
            self.values
                .iter()
                .copied()
                .flat_map(|f| repeat(f).take(shape.1)),
        )
        .take(shape.0)
        .flatten()
        .collect_matrix((self.shape.0 * shape.0, self.shape.1 * shape.1))
    }
}

impl<F: Float> IntoIterator for Matrix<F> {
    type Item = F;
    type IntoIter = <Vec<F> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

pub trait CollectMatrix<F> {
    fn collect_matrix(self, shape: (usize, usize)) -> Matrix<F>;
}

impl<F: Float, I: IntoIterator<Item = F>> CollectMatrix<F> for I {
    fn collect_matrix(self, shape: (usize, usize)) -> Matrix<F> {
        Matrix::new(self.into_iter().collect(), shape)
    }
}

pub(crate) fn meshgrid<F: Float, I, I2>(u: I, v: I2) -> (Matrix<F>, Matrix<F>)
where
    I: IntoIterator<Item = F>,
    I::IntoIter: Clone + ExactSizeIterator,
    I2: IntoIterator<Item = F>,
    I2::IntoIter: ExactSizeIterator,
{
    let u = u.into_iter();
    let v = v.into_iter();
    let shape = (v.len(), u.len());
    let values = u.cycle().take(shape.0 * shape.1).collect();
    let values2 = v.flat_map(|v| repeat(v).take(shape.1)).collect();

    (
        Matrix { values, shape },
        Matrix {
            values: values2,
            shape,
        },
    )
}

/// Similar to numpy.interp
pub(crate) fn interp<F>(
    x_input: impl IntoIterator<Item = F>,
    coords: impl IntoIterator<Item = (F, F)> + Clone,
) -> Vec<F>
where
    F: Add + Float + Mul + Sub,
{
    x_input
        .into_iter()
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
        interp(x, xs.into_iter().zip(ys.into_iter())),
        vec![3., 3., 2.5, 0.5599999999999996, 0.]
    );

    let x = [2.5, -1., 7.5];
    let xs = [0., 1., 2., 3., 4.5];
    let ys = [0., 2., 5., 3., 2.];

    assert_eq!(
        interp(x, xs.into_iter().zip(ys.into_iter())),
        vec![4., 0., 2.]
    );
}

#[test]
fn test_meshgrid() {
    let (m1, m2) = meshgrid([1., 2., 3.], [4., 5., 8., 7.]);

    assert_eq!(
        m1.values,
        vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]
    );
    assert_eq!(m1.shape, (4, 3));
    assert_eq!(
        m2.values,
        vec![4., 4., 4., 5., 5., 5., 8., 8., 8., 7., 7., 7.]
    );
    assert_eq!(m2.shape, (4, 3));
}

#[test]
fn test_tile() {
    let m1 = Matrix {
        values: vec![1., 2., 3., 4., 5.],
        shape: (5, 1),
    };
    let m2 = m1.tile((2, 3));

    assert_eq!(
        m2.values,
        vec![
            1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 1., 1., 1., 2., 2., 2., 3.,
            3., 3., 4., 4., 4., 5., 5., 5.
        ]
    );
    assert_eq!(m2.shape(), (10, 3));

    let m1 = Matrix {
        values: vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.],
        shape: (5, 2),
    };
    let m2 = m1.tile((2, 2));

    assert_eq!(
        m2.values,
        vec![
            1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4., 5., 5., 5., 5., 1., 1.,
            1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4., 5., 5., 5., 5.
        ]
    );
    assert_eq!(m2.shape(), (10, 4));
}
