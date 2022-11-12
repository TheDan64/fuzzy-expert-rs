use std::collections::HashMap;

use crate::variable::{Variable, VariableKey};

// TODO: Might map to Fact eventually?
#[derive(Debug)]
pub struct Outputs {
    defuzzificated_inferred_memberships: HashMap<VariableKey, f64>,
    inferred_cf: f64,
}

impl Outputs {
    pub(crate) fn new(defuzzificated_inferred_memberships: HashMap<VariableKey, f64>, inferred_cf: f64) -> Self {
        Self {
            defuzzificated_inferred_memberships,
            inferred_cf,
        }
    }

    pub fn get_inferred_membership<I>(&self, var: Variable<I>) -> Option<f64> {
        self.defuzzificated_inferred_memberships.get(&var.0).copied()
    }

    pub fn inferred_cf(&self) -> f64 {
        self.inferred_cf
    }
}
