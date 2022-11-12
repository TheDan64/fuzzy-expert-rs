mod dsl;
pub mod inference;
pub mod inputs;
mod linspace;
mod math;
pub mod ops;
pub mod outputs;
pub mod rules;
pub mod terms;
pub mod variable;

pub mod prelude {
    pub use crate::inference::DecompInference;
    pub use crate::ops::{AndOp, CompositionOp, DefuzzificationOp, ImplicationOp, OrOp, ProductionLink};
    pub use crate::outputs::Outputs;
    pub use crate::rules::Rules;
    pub use crate::terms::{Term, Terms};
    pub use crate::variable::{Variable, VariableKey};
}

/// A value between zero and one
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Probability(f64);

impl TryFrom<f64> for Probability {
    type Error = ();

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if (0.0..=1.).contains(&value) {
            Ok(Probability(value))
        } else {
            Err(())
        }
    }
}

enum Fact {
    Crisp(f64),
    Fuzzy(Vec<(f64, Probability)>),
}
