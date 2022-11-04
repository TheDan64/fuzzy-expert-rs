use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};

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

// REVIEW: Is this really only input variables?
#[derive(Default)]
pub struct Variables<T>(SlotMap<VariableKey, VariableContraints>, PhantomData<T>);

impl<T: Terms> Variables<T> {
    pub fn new() -> Self {
        Self(SlotMap::with_key(), PhantomData)
    }

    pub fn add<I: Into<T>>(
        &mut self,
        bounds: impl RangeBounds<f32>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> Variable<I> {
        let key = self.0.insert(VariableContraints {
            universe: (bounds.start_bound().cloned(), bounds.end_bound().cloned()),
            certainty_factor: certainty_factor.into().unwrap_or(ZeroOne(1.)),
        });

        Variable(key, PhantomData)
    }
}

struct VariableContraints {
    universe: (Bound<f32>, Bound<f32>),
    certainty_factor: ZeroOne,
}

#[derive(Default)]
pub struct Rules<T, O>(Vec<Rule<T, O>>);

impl<T: Terms, O> Rules<T, O> {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn add(&mut self, premise: Expr<T>, consequence: O)
    where
        O: Into<T>,
    {
        self.0.push(Rule {
            premise,
            consequence,
        });
    }
}

struct Rule<T, O> {
    premise: Expr<T>,
    consequence: O,
}

pub trait Terms {
    fn values(&self) -> &'static [(f32, f32)];
}

#[derive(Default)]
pub struct Inputs(HashMap<VariableKey, f32>);

impl Inputs {
    pub fn new() -> Self {
        Inputs(HashMap::new())
    }

    pub fn add<I>(&mut self, var: Variable<I>, val: f32) {
        self.0.insert(var.0, val);
    }
}

fn eval<'r, T: Terms, O: Into<T>>(
    vars: &Variables<T>,
    rules: &'r Rules<T, O>,
    inputs: &Inputs,
) -> &'r O {
    // Placeholder
    &rules.0[0].consequence
}

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

    enum Decision {
        Approve,
        Reject,
    }

    enum VarTerms {
        Score(Score),
        Ratio(Ratio),
        Credit(Credit),
        Decision(Decision),
    }

    impl From<Decision> for VarTerms {
        fn from(d: Decision) -> Self {
            Self::Decision(d)
        }
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
                Self::Decision(Decision::Approve) => &[(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
                Self::Decision(Decision::Reject) => &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
            }
        }
    }

    let mut vars = Variables::<VarTerms>::new();
    let score = vars.add::<Score>(150. ..=200., ZeroOne(1.));
    let ratio = vars.add::<Ratio>(0.1..=1., ZeroOne(1.));
    let credit = vars.add::<Credit>(0. ..=10., ZeroOne(1.));
    // let decision = vars.add::<Decision>(0. ..=10., ZeroOne(1.));
    let mut rules = Rules::new();

    rules.add(
        score
            .eq(Score::High)
            .and2(ratio.eq(Ratio::Good), credit.eq(Credit::Good)),
        Decision::Approve,
    );
    rules.add(
        score
            .eq(Score::Low)
            .and(ratio.eq(Ratio::Bad).or(credit.eq(Credit::Bad))),
        Decision::Reject,
    );

    let mut inputs = Inputs::new();

    inputs.add(score, 190.);
    inputs.add(ratio, 0.39);
    inputs.add(credit, 1.5);

    eval(&vars, &rules, &inputs);
}
