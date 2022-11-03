use crate::Variable;

pub(crate) enum Expr {
    Eq(Variable, &'static str),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}
