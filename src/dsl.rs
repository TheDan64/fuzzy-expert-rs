use crate::{Variable, VariableKey};

// TODO: Support modifiers
pub enum Expr<T> {
    Is(VariableKey, T),
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

    pub fn propositions(&self) -> Vec<(&VariableKey, &T, &[()])> {
        let mut props = Vec::new();

        fn parse<'p, T>(expr: &'p Expr<T>, out: &mut Vec<(&'p VariableKey, &'p T, &'p [()])>) {
            match expr {
                Expr::Is(var_key, term) => out.push((var_key, term, &[])),
                Expr::And(exprs) | Expr::Or(exprs) => {
                    for expr in exprs {
                        parse(expr, out);
                    }
                }
            }
        }

        parse(self, &mut props);

        props
    }
}

impl<I> Variable<I> {
    pub fn is<T>(self, rhs: I) -> Expr<T>
    where
        I: Into<T>,
    {
        Expr::Is(self.0, rhs.into())
    }
}
