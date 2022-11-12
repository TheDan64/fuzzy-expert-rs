use std::collections::HashMap;

use crate::variable::{Variable, VariableKey};

#[derive(Default)]
pub struct Inputs(pub(crate) HashMap<VariableKey, f64>);

impl Inputs {
    pub fn new() -> Self {
        Inputs(HashMap::new())
    }

    // TODO: K: VariableKind {Crisp, Fuzzy}, val: K::Value {f64, Vec<(f64, f64)>}
    pub fn add<I>(&mut self, var: Variable<I>, val: f64) {
        self.0.insert(var.0, val);
    }
}
