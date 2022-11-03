use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

mod dsl;

use dsl::Expr;

/// A value between zero and one
struct ZeroOne(f32);

#[derive(Clone)]
struct Variable(Arc<VariableInner>);

impl Variable {
    pub fn new(
        universe_range: Range<f32>,
        terms: HashMap<&'static str, Vec<(f32, f32)>>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> Self {
        Self(Arc::new(VariableInner {
            universe_range,
            terms,
            certainty_factor: certainty_factor.into().unwrap_or(ZeroOne(1.)),
        }))
    }
}

struct VariableInner {
    universe_range: Range<f32>,
    terms: HashMap<&'static str, Vec<(f32, f32)>>,
    certainty_factor: ZeroOne,
}

struct Rule {
    premise: Expr,
    consequence: (),
}

fn run(rules: &[Rule], input: &[()]) {}

#[test]
fn test_bank_loan() {
    macro_rules! map(
        { $($key:expr => $value:expr,)+ } => {
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
    let score = Variable::new(150. ..200., score_terms, ZeroOne(1.));
    let ratio = Variable::new(0.1..1., ratio_terms, ZeroOne(1.));
    let credit = Variable::new(0. ..10., credit_terms, ZeroOne(1.));
    // let vars: HashMap<_, _> = [("score", score), ("ratio", ratio), ("credit", credit)]
    //     .into_iter()
    //     .collect();
    let rule1 = Rule {
        premise: Expr::And(vec![
            Expr::Eq(score.clone(), "High"),
            Expr::Eq(ratio.clone(), "Goodr"),
            Expr::Eq(credit.clone(), "Goodc"),
        ]),
        consequence: (),
    };
    let rule2 = Rule {
        premise: Expr::And(vec![
            Expr::Eq(score, "Low"),
            Expr::Or(vec![Expr::Eq(ratio, "Badr"), Expr::Eq(credit, "Badc")]),
        ]),
        consequence: (),
    };

    run(&[rule1, rule2], &[]);
}
