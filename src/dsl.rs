use crate::Variable;

pub enum Expr<T> {
    Eq(Variable, T),
    And(Vec<Expr<T>>),
    Or(Vec<Expr<T>>),
}

impl<T> Expr<T> {
    pub fn or(self, rhs: Expr<T>) -> Self {
        Expr::Or(vec![self, rhs])
    }

    pub fn and(self, rhs: Expr<T>) -> Self {
        Expr::And(vec![self, rhs])
    }

    pub fn and2(self, rhs: Expr<T>, rhs2: Expr<T>) -> Self {
        Expr::And(vec![self, rhs, rhs2])
    }
}

impl Variable {
    pub fn eq<T>(self, rhs: impl Into<T>) -> Expr<T> {
        Expr::Eq(self, rhs.into())
    }
}
