use std::marker::PhantomData;
use std::ops::{Range, RangeBounds};

use slotmap::{new_key_type, SlotMap};

mod dsl;

use dsl::Expr;

/// A value between zero and one
pub struct ZeroOne(f32);

new_key_type! {
    /// A variable
    pub struct VariableKey;
}

pub struct Variable<I>(VariableKey, PhantomData<I>);

impl<I> Clone for Variable<I> {
    fn clone(&self) -> Self {
        Variable(self.0, PhantomData)
    }
}

impl<I> Copy for Variable<I> {}

#[derive(Default)]
pub struct Variables<T>(SlotMap<VariableKey, VariableContraints>, PhantomData<T>);

impl<T: Terms> Variables<T> {
    pub fn new() -> Self {
        Self(SlotMap::with_key(), PhantomData)
    }

    pub fn add<I: Into<T>>(
        &mut self,
        universe: Range<f32>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> Variable<I> {
        let key = self.0.insert(VariableContraints {
            universe,
            certainty_factor: certainty_factor.into().unwrap_or(ZeroOne(1.)),
        });

        Variable(key, PhantomData)
    }
}

struct VariableContraints {
    universe: Range<f32>,
    certainty_factor: ZeroOne,
}

#[derive(Default)]
pub struct Rules<T>(Vec<Rule<T>>);

impl<T: Terms> Rules<T> {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn add(&mut self, premise: Expr<T>) {
        self.0.push(Rule {
            premise,
            consequence: (),
        });
    }
}

struct Rule<T> {
    premise: Expr<T>,
    consequence: (),
}

pub trait Terms {
    fn values(&self) -> &'static [(f32, f32)];
}

fn eval<T: Terms>(vars: &Variables<T>, rules: &Rules<T>, input: &[()]) {}

#[test]
fn test_bank_loan() {
    enum Score {
        High,
        Low,
    }

    enum Ratio {
        Good,
        Bad,
    }

    enum Credit {
        Good,
        Bad,
    }

    enum VarTerms {
        Score(Score),
        Ratio(Ratio),
        Credit(Credit),
    }

    impl From<Score> for VarTerms {
        fn from(s: Score) -> Self {
            Self::Score(s)
        }
    }

    impl From<Ratio> for VarTerms {
        fn from(r: Ratio) -> Self {
            Self::Ratio(r)
        }
    }

    impl From<Credit> for VarTerms {
        fn from(c: Credit) -> Self {
            Self::Credit(c)
        }
    }

    impl Terms for VarTerms {
        fn values(&self) -> &'static [(f32, f32)] {
            match self {
                Self::Score(Score::High) => &[(175., 0.), (180., 0.2), (185., 0.7), (190., 1.)],
                Self::Score(Score::Low) => &[
                    (155., 1.),
                    (160., 0.8),
                    (165., 0.5),
                    (170., 0.2),
                    (175., 0.),
                ],
                Self::Ratio(Ratio::Good) => &[(0.3, 1.), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)],
                Self::Ratio(Ratio::Bad) => &[(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)],
                Self::Credit(Credit::Good) => &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
                Self::Credit(Credit::Bad) => &[(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
            }
        }
    }

    let mut vars = Variables::<VarTerms>::new();
    let score = vars.add::<Score>(150. ..200., ZeroOne(1.));
    let ratio = vars.add::<Ratio>(0.1..1., ZeroOne(1.));
    let credit = vars.add::<Credit>(0. ..10., ZeroOne(1.));
    let mut rules = Rules::new();

    rules.add(
        score
            .eq(Score::High)
            .and2(ratio.eq(Ratio::Good), credit.eq(Credit::Good)),
    );
    rules.add(
        score
            .eq(Score::Low)
            .and(ratio.eq(Ratio::Bad).or(credit.eq(Credit::Bad))),
    );

    eval(&vars, &rules, &[]);
}
