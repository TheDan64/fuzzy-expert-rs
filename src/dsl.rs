use crate::Variable;

pub enum Expr {
    // REVIEW: Maybe fixed_map crate for RHS? Could be Key or Value? Or Just an enum?
    Eq(Variable, &'static str),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}

impl Expr {
    pub fn or(self, rhs: Expr) -> Self {
        Expr::Or(vec![self, rhs])
    }

    pub fn and(self, rhs: Expr) -> Self {
        Expr::And(vec![self, rhs])
    }

    pub fn and2(self, rhs: Expr, rhs2: Expr) -> Self {
        Expr::And(vec![self, rhs, rhs2])
    }
}

impl Variable {
    pub fn eq(self, rhs: &'static str) -> Expr {
        Expr::Eq(self, rhs)
    }
}
