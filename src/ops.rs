use std::iter::Sum;
use std::ops::AddAssign;

use num::Float;

use crate::math::interp;

/// And operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
pub enum AndOp {
    Min,
    Prod,
    BoundedProd,
    DrasticProd,
}

impl AndOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        match self {
            Self::Min => ProductionLink::Min.call(u, v),
            Self::Prod => ProductionLink::Prod.call(u, v),
            Self::BoundedProd => ProductionLink::BoundedProd.call(u, v),
            Self::DrasticProd => ProductionLink::DrasticProd.call(u, v),
        }
    }
}

/// Or operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
pub enum OrOp {
    Max,
    ProbOr,
    BoundedSum,
    DrasticSum,
}

impl OrOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        match self {
            Self::Max => ProductionLink::Max.call(u, v),
            Self::ProbOr => ProductionLink::ProbOr.call(u, v),
            Self::BoundedSum => ProductionLink::BoundedSum.call(u, v),
            Self::DrasticSum => ProductionLink::DrasticSum.call(u, v),
        }
    }
}

pub enum CompositionOp {
    MaxMin,
    MaxProd,
}

/// Implication operator method for computing the compisitions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
pub enum ImplicationOp {
    Ra,
    Rm,
    Rc,
    Rb,
    Rs,
    Rg,
    Rsg,
    Rgs,
    Rgg,
    Rss,
}

impl ImplicationOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        u.into_iter().zip(v.into_iter()).map(move |(u, v)| match self {
            Self::Ra => F::min(F::one(), F::one() - u + v),
            Self::Rm => F::max(F::min(u, v), F::one() - u),
            Self::Rc => F::min(u, v),
            Self::Rb => F::max(F::one() - u, v),
            Self::Rs => {
                if u <= v {
                    F::one()
                } else {
                    F::zero()
                }
            },
            Self::Rg => {
                if u <= v {
                    F::one()
                } else {
                    v
                }
            },
            Self::Rsg => F::min(
                Self::Rs.call(Some(u), Some(v)).into_iter().next().expect("unreachable"),
                Self::Rg
                    .call(Some(F::one() - u), Some(F::one() - v))
                    .into_iter()
                    .next()
                    .expect("unreachable"),
            ),
            Self::Rgs => F::min(
                Self::Rg.call(Some(u), Some(v)).into_iter().next().expect("unreachable"),
                Self::Rs
                    .call(Some(F::one() - u), Some(F::one() - v))
                    .into_iter()
                    .next()
                    .expect("unreachable"),
            ),
            Self::Rgg => F::min(
                Self::Rg.call(Some(u), Some(v)).into_iter().next().expect("unreachable"),
                Self::Rg
                    .call(Some(F::one() - u), Some(F::one() - v))
                    .into_iter()
                    .next()
                    .expect("unreachable"),
            ),
            Self::Rss => F::min(
                Self::Rs.call(Some(u), Some(v)).into_iter().next().expect("unreachable"),
                Self::Rs
                    .call(Some(F::one() - u), Some(F::one() - v))
                    .into_iter()
                    .next()
                    .expect("unreachable"),
            ),
        })
    }
}

/// Method for aggregating the consequences of the fuzzy rules
#[derive(Clone, Copy, Debug)]
pub enum ProductionLink {
    Min,
    Prod,
    BoundedProd,
    DrasticProd,
    Max,
    ProbOr,
    BoundedSum,
    DrasticSum,
}

impl ProductionLink {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        u.into_iter().zip(v.into_iter()).map(move |(u, v)| match self {
            Self::Max => F::max(u, v),
            Self::ProbOr => u + v - u * v,
            Self::BoundedSum => F::min(F::one(), u + v),
            Self::DrasticSum => {
                if v == F::zero() {
                    u
                } else if u == F::zero() {
                    v
                } else {
                    F::one()
                }
            },
            Self::Min => F::min(u, v),
            Self::Prod => u * v,
            Self::BoundedProd => F::max(F::zero(), u + v - F::one()),
            Self::DrasticProd => {
                if v == F::zero() {
                    u
                } else if u == F::one() {
                    v
                } else {
                    F::zero()
                }
            },
        })
    }
}

/// Method for defuzzifcating the resulting membership function.
#[derive(Clone, Copy, Debug)]
pub enum DefuzzificationOp {
    /// Center of Gravity
    Cog,
    /// Bisector of Area
    Boa,
    /// Mean of the values for which the membership function is maximum
    Mom,
    /// Largest value for which the membership function is maximum
    Lom,
    /// Smallest value for which the membership function is minimum
    Som,
}

impl DefuzzificationOp {
    pub fn call<F: Float + Sum + AddAssign>(self, universe: &[F], membership: &[F]) -> F {
        match self {
            Self::Cog => {
                let n_areas = universe.len() - 1;
                let mut areas = Vec::with_capacity(n_areas);
                let mut centroids = Vec::with_capacity(n_areas);
                let two = F::one() + F::one();
                let three = two + F::one();

                for i in 0..n_areas {
                    let base = universe[i + 1] - universe[i];
                    let area_rect = F::min(membership[i], membership[i + 1]) * base;
                    let center_rect = universe[i] + base / two;
                    let (area_tria, center_tri) = if membership[i + 1] == membership[i] {
                        (F::zero(), F::zero())
                    } else if membership[i + 1] > membership[i] {
                        (
                            base * F::abs(membership[i + 1] - membership[i]) / two,
                            universe[i] + two / three * base,
                        )
                    } else {
                        (
                            base * F::abs(membership[i + 1] - membership[i]) / two,
                            universe[i] + F::one() / three * base,
                        )
                    };
                    let area = area_rect + area_tria;
                    let center = if area == F::zero() {
                        F::zero()
                    } else {
                        (area_rect * center_rect + area_tria * center_tri) / (area_rect + area_tria)
                    };

                    areas.push(area);
                    centroids.push(center);
                }

                let den = areas.iter().copied().sum::<F>();
                let num = areas
                    .into_iter()
                    .zip(centroids.into_iter())
                    .map(|(area, cent)| area * cent)
                    .sum::<F>();

                num / den
            },
            Self::Boa => {
                let n_areas = universe.len() - 1;
                let mut areas = Vec::with_capacity(n_areas);

                for i_area in 0..n_areas {
                    let base = universe[i_area + 1] - universe[i_area];
                    let area = (membership[i_area] + membership[i_area + 1]) * base / F::from(2.).expect("unreachable");
                    areas.push(area);
                }

                let total_area = areas.iter().copied().sum::<F>();
                let target = total_area / F::from(2.).expect("unreachable");
                let mut cum_area = F::zero();
                let mut i_area = 0;

                for i in 0..=n_areas {
                    cum_area += areas[i];
                    i_area = i;
                    if cum_area >= target {
                        break;
                    }
                }

                let xp = [universe[i_area], universe[i_area + 1]];
                let fp = [cum_area - areas[i_area], cum_area];

                interp(Some(target), xp.into_iter().zip(fp.into_iter()))
                    .into_iter()
                    .next()
                    .expect("unreachable")
            },
            Self::Mom => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                let (len, sum) = universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .enumerate()
                    .fold((0usize, F::zero()), |(_i, accum), (i, next)| (i + 1, accum + next));

                sum / F::from(len).unwrap()
            },
            Self::Lom => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .reduce(F::max)
                    .unwrap()
            },
            Self::Som => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .reduce(F::min)
                    .unwrap()
            },
        }
    }
}
