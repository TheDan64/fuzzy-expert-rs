use std::collections::HashMap;
use std::ops::Range;

mod dsl;

use dsl::Expr;

struct Variable {
    universe_range: Range<f32>,
    terms: HashMap<String, Vec<(f32, f32)>>,
}

struct Rule {
    premise: Expr,
    consequence: (),
}

#[test]
fn test_bank_loan() {
    let score = Variable {
        universe_range: 150. ..200.,
        terms: HashMap::new(),
    };
    let vars: HashMap<_, _> = [("score", score)].into_iter().collect();
}
