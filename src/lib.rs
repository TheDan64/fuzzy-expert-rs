use std::collections::HashMap;
use std::ops::Range;

use slotmap::{DefaultKey, SlotMap};

mod dsl;

use dsl::Expr;

// TODO: FuzzyKey

/// A value between zero and one
pub struct ZeroOne(f32);

pub struct Variables(SlotMap<DefaultKey, Variable>);

impl Variables {
    pub fn new() -> Self {
        Self(SlotMap::new())
    }

    pub fn add(
        &mut self,
        universe_range: Range<f32>,
        terms: HashMap<&'static str, Vec<(f32, f32)>>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> DefaultKey {
        self.0.insert(Variable {
            universe_range,
            terms,
            certainty_factor: certainty_factor.into().unwrap_or(ZeroOne(1.)),
        })
    }
}

struct Variable {
    universe_range: Range<f32>,
    terms: HashMap<&'static str, Vec<(f32, f32)>>,
    certainty_factor: ZeroOne,
}

pub struct Rules(Vec<Rule>);

impl Rules {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn add(&mut self, premise: Expr) {
        self.0.push(Rule {
            premise,
            consequence: (),
        });
    }
}

struct Rule {
    premise: Expr,
    consequence: (),
}

fn eval(vars: &Variables, rules: &Rules, input: &[()]) {}

#[test]
fn test_bank_loan() {
    macro_rules! map(
        ($($key:expr => $value:expr,)+) => {
            {
                let mut m = ::std::collections::HashMap::with_capacity([$($key,)+].len());
                $(
                    m.insert($key, $value);
                )+
                m
            }
         };
    );

    let score_terms = map! {
        "High" => vec![(175., 0.), (180., 0.2), (185., 0.7), (190., 1.)],
        "Low" => vec![(155., 1.), (160., 0.8), (165., 0.5), (170., 0.2), (175., 0.)],
    };
    let ratio_terms = map! {
        "Goodr" => vec![(0.3, 1.), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)],
        "Badr" => vec![(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)],
    };
    let credit_terms = map! {
        "Goodc" => vec![(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
        "Badc" => vec![(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
    };
    let mut vars = Variables::new();
    let score = vars.add(150. ..200., score_terms, ZeroOne(1.));
    let ratio = vars.add(0.1..1., ratio_terms, ZeroOne(1.));
    let credit = vars.add(0. ..10., credit_terms, ZeroOne(1.));
    let mut rules = Rules::new();

    rules.add(Expr::And(vec![
        Expr::Eq(score, "High"),
        Expr::Eq(ratio, "Goodr"),
        Expr::Eq(credit, "Goodc"),
    ]));
    rules.add(Expr::And(vec![
        Expr::Eq(score, "Low"),
        Expr::Or(vec![Expr::Eq(ratio, "Badr"), Expr::Eq(credit, "Badc")]),
    ]));

    eval(&vars, &rules, &[]);
}
