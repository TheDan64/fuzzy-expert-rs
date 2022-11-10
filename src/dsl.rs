use crate::{Variable, VariableKey};

// TODO: Support modifiers
// TODO: Return to And(Vec<Expr<T>>), Or(Vec<Expr<T>>),
// for simplicity and possibly cache friendliness
pub enum Expr<T> {
    Is(VariableKey, T),
    And(Box<Expr<T>>, Box<Expr<T>>),
    Or(Box<Expr<T>>, Box<Expr<T>>),
}

impl<T> Expr<T> {
    pub fn or(self, rhs: Expr<T>) -> Self {
        Expr::Or(Box::new(self), Box::new(rhs))
    }

    pub fn and(self, rhs: Expr<T>) -> Self {
        Expr::And(Box::new(self), Box::new(rhs))
    }

    pub fn and2(self, rhs: Expr<T>, rhs2: Expr<T>) -> Self {
        self.and(rhs.and(rhs2))
    }

    pub(crate) fn propositions(&self) -> Vec<(VariableKey, &T, &[()])> {
        let mut props = Vec::new();

        fn parse<'p, T>(expr: &'p Expr<T>, out: &mut Vec<(VariableKey, &'p T, &'p [()])>) {
            match expr {
                Expr::Is(var_key, term) => out.push((*var_key, term, &[])),
                Expr::And(expr, expr2) | Expr::Or(expr, expr2) => {
                    parse(expr, out);
                    parse(expr2, out)
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
