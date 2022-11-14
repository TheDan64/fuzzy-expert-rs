use crate::dsl::Expr;

#[derive(Default)]
pub struct Rules<T>(pub(crate) Vec<Rule<T>>);

impl<T> Rules<T> {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Rules(Vec::with_capacity(capacity))
    }

    // REVIEW: Maybe consequence should be (Variable<I>, T) so we can manually
    // turn it into Expr::Is(VariableKey, T)?
    pub fn add(&mut self, premise: Expr<T>, consequence: Expr<T>) {
        self.0.push(Rule {
            premise,
            consequence,
            // TODO: CFs should be overridable
            cf: 1.0,
            threshold_cf: 0.,
        });
    }
}

pub(crate) struct Rule<T> {
    // TODO: Rename to condition?
    pub(crate) premise: Expr<T>,
    // TODO: Rename to result or output?
    pub(crate) consequence: Expr<T>,
    pub(crate) cf: f64,
    pub(crate) threshold_cf: f64,
}
