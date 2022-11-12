mod dsl;
pub mod inference;
pub mod inputs;
mod linspace;
mod math;
pub mod ops;
pub mod outputs;
pub mod rules;
pub mod variable;

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
