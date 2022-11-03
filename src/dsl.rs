use slotmap::DefaultKey;

pub enum Expr {
    // REVIEW: Maybe fixed_map crate for RHS? Could be Key or Value? Or Just an enum?
    Eq(DefaultKey, &'static str),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}
