pub enum Expr {
    Eq(String, ()),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}
