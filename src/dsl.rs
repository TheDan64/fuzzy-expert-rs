use crate::{Variable, VariableKey};

pub enum Expr<T> {
    Eq(VariableKey, T),
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

impl<I> Variable<I> {
    pub fn eq<T>(self, rhs: I) -> Expr<T>
    where
        I: Into<T>,
    {
        Expr::Eq(self.0, rhs.into())
    }
}
