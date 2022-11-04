use std::collections::HashMap;
use std::ops::{Range, RangeBounds};

use slotmap::{new_key_type, SlotMap};

mod dsl;

use dsl::Expr;

/// A value between zero and one
pub struct ZeroOne(f32);

new_key_type! {
    /// A variable
    pub struct Variable;
}

#[derive(Default)]
pub struct Variables(SlotMap<Variable, VariableContraints>);

impl Variables {
    pub fn new() -> Self {
        Self(SlotMap::with_key())
    }

    pub fn add(
        &mut self,
        universe: Range<f32>,
        terms: HashMap<&'static str, Vec<(f32, f32)>>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> Variable {
        self.0.insert(VariableContraints {
            universe,
            terms,
            certainty_factor: certainty_factor.into().unwrap_or(ZeroOne(1.)),
        })
    }
}

struct VariableContraints {
    universe: Range<f32>,
    terms: HashMap<&'static str, Vec<(f32, f32)>>,
    certainty_factor: ZeroOne,
}

#[derive(Default)]
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

    rules.add(score.eq("High").and2(ratio.eq("Goodr"), credit.eq("Goodc")));
    rules.add(score.eq("Low").and(ratio.eq("Badr").or(credit.eq("Badc"))));

    eval(&vars, &rules, &[]);
}
