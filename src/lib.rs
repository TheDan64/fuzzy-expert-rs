use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use slotmap::{new_key_type, SlotMap};

mod dsl;
mod linspace;
mod math;

use dsl::Expr;
use linspace::Linspace;
use math::interp;

/// A value between zero and one
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct ZeroOne(f32);

new_key_type! {
    /// A variable key
    pub struct VariableKey;
}

pub struct Variable<I>(VariableKey, PhantomData<I>);

impl<I> Clone for Variable<I> {
    fn clone(&self) -> Self {
        Variable(self.0, PhantomData)
    }
}

impl<I> Copy for Variable<I> {}

// REVIEW: Is this really only input variables?
#[derive(Default)]
pub struct Variables<T>(SlotMap<VariableKey, VariableContraints<T>>);

impl<T: Terms> Variables<T> {
    pub fn new() -> Self {
        Self(SlotMap::with_key())
    }

    pub fn add<I: Into<T>>(
        &mut self,
        universe_range: RangeInclusive<f32>,
        certainty_factor: impl Into<Option<ZeroOne>>,
    ) -> Variable<I> {
        let key = self.0.insert(VariableContraints::new(
            universe_range,
            certainty_factor.into().unwrap_or(ZeroOne(1.)),
        ));

        Variable(key, PhantomData)
    }
}

struct VariableContraints<T> {
    universe: Vec<f32>,
    certainty_factor: ZeroOne,
    min_u: f32,
    max_u: f32,
    terms: HashMap<T, Vec<f32>>,
}

impl<T: Terms> VariableContraints<T> {
    pub fn new(universe_range: RangeInclusive<f32>, certainty_factor: ZeroOne) -> Self {
        // self.terms: dict = terms

        // REVIEW: This is an opt param in the original
        let step = 0.1;
        let min_u = *universe_range.start();
        let max_u = *universe_range.end();
        // floor is closest approx to what python does for int() conversion. But at least one edgecase exists
        // where the decimals are really long: int(4.999999999999999999) == 5
        let num = ((max_u - min_u) / step).floor() as usize + 1;
        let universe = Linspace::new(min_u, max_u, num).collect();
        let start_terms = T::build();
        let mut this = Self {
            universe,
            certainty_factor,
            min_u,
            max_u,
            terms: HashMap::with_capacity(start_terms.len()),
        };

        // Load from tuple?
        if false {
            unimplemented!();
        // Load from list
        } else {
            for (term, membership) in &start_terms {
                let xp = membership.iter().map(|(xp, _)| *xp);
                this.add_points_to_universe(xp);
                this.terms
                    .insert(*term, interp(&this.universe, membership.iter().copied()));
            }
        }

        this
    }

    fn add_points_to_universe(&mut self, points: impl IntoIterator<Item = f32>) {
        // Adds new points to the universe
        let iter = points.into_iter().map(|p| p.clamp(self.min_u, self.max_u));
        // REVIEW: This should probably work on a copy of universe
        self.universe.extend(iter);
        self.universe
            .sort_unstable_by(|a, b| a.partial_cmp(b).expect("not to find unsortable floats"));
        self.universe.dedup();

        // Expand existent membership functions with the new points

        // Update the universe with the new points
        // self.universe = universe;
    }
}

#[derive(Default)]
pub struct Rules<T, O>(Vec<Rule<T, O>>);

impl<T: Terms, O> Rules<T, O> {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn add(&mut self, premise: Expr<T>, consequence: O)
    where
        O: Into<T>,
    {
        self.0.push(Rule {
            premise,
            consequence,
        });
    }
}

struct Rule<T, O> {
    // TODO: Rename to condition
    premise: Expr<T>,
    // TODO: Rename to result or output
    consequence: O,
}

pub trait Terms: Copy + Eq + Hash + Sized {
    // fn values(&self) -> &'static [(f32, f32)];
    fn build() -> HashMap<Self, Vec<(f32, f32)>>;
}

#[derive(Default)]
pub struct Inputs(HashMap<VariableKey, f32>);

impl Inputs {
    pub fn new() -> Self {
        Inputs(HashMap::new())
    }

    // TODO: K: VariableKind {Crisp, Fuzzy}, val: K::Value {f32, Vec<(f32, f32)>}
    pub fn add<I>(&mut self, var: Variable<I>, val: f32) {
        self.0.insert(var.0, val);
    }
}

enum Fact {
    Crisp(f32),
    Fuzzy(Vec<(f32, ZeroOne)>),
}

/// And operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
pub enum AndOp {
    Min,
    Prod,
    BoundedProd,
    DrasticProd,
}

/// Or operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
pub enum OrOp {
    Max,
    ProbOr,
    BoundedSum,
    DrasticSum,
}

pub enum CompositionOp {
    MaxMin,
    MaxProd,
}

/// Implication operator method for computing the compisitions of propositions
/// in a fuzzy rule premise.
pub enum ImplicationOp {
    Ra,
    Rm,
    Rc,
    Rb,
    Rs,
    Rg,
    Rsg,
    Rgs,
    Rgg,
    Rss,
}

/// Method for aggregating the consequences of the fuzzy rules
pub enum ProductionLink {
    Min,
    Prod,
    BoundedProd,
    DrasticProd,
    Max,
    ProbOr,
    BoundedSum,
    DrasticSum,
}

/// Method for defuzzifcating the resulting membership function.
pub enum DefuzzificationOp {
    /// Center of Gravity
    Cog,
    /// Bisector of Area
    Boa,
    /// Mean of the values for which the membership function is maximum
    Mom,
    /// Largest value for which the membership function is maximum
    Lom,
    /// Smallest value for which the membership function is minimum
    Som,
}

pub struct DecompInference {
    and_op: AndOp,
    or_op: OrOp,
    comp_op: CompositionOp,
    imp_op: ImplicationOp,
    prod_link: ProductionLink,
    defuzz_op: DefuzzificationOp,
}

impl DecompInference {
    pub fn new(
        and_op: AndOp,
        or_op: OrOp,
        comp_op: CompositionOp,
        imp_op: ImplicationOp,
        prod_link: ProductionLink,
        defuzz_op: DefuzzificationOp,
    ) -> Self {
        Self {
            and_op,
            or_op,
            comp_op,
            imp_op,
            prod_link,
            defuzz_op,
        }
    }
    pub fn eval<'r, T: Terms, O: Into<T>>(
        &self,
        vars: &mut Variables<T>,
        rules: &'r Rules<T, O>,
        inputs: &Inputs,
    ) -> &'r O {
        // Convert Inputs to facts
        // Converts input values to FIS facts
        let mut fact_values = HashMap::with_capacity(inputs.0.len());
        let mut fact_cf = HashMap::with_capacity(inputs.0.len());

        for (key, input_value) in &inputs.0 {
            fact_values.insert(*key, *input_value);
            fact_cf.insert(*key, vars.0[*key].certainty_factor);
        }

        // Fuzzificate Facts
        // Convert crisp facts to membership functions
        let mut fact_types = HashMap::with_capacity(fact_values.len());

        for (key, fact_value) in fact_values {
            // py: (float, int?)
            // This will eventually be an enum match
            if true {
                // Fuzzificate Crisp Fact
                vars.0[key].add_points_to_universe(Some(fact_value));
                // self.fact_values[fact_name] = np.array(
                //     [1 if u == fact_value else 0 for u in self.variables[fact_name].universe]
                // )
                fact_types.insert(key, "Crisp");
            // py: list
            } else {
                // TODO: Fuzzificate fuzzy fact
                fact_types.insert(key, "Fuzzy");
                unimplemented!();
            }
        }

        // Placeholder
        &rules.0[0].consequence
    }
}

#[test]
fn test_bank_loan() {
    #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum Score {
        High,
        Low,
    }

    #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum Ratio {
        Good,
        Bad,
    }

    #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum Credit {
        Good,
        Bad,
    }

    #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum Decision {
        Approve,
        Reject,
    }

    #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum VarTerms {
        Score(Score),
        Ratio(Ratio),
        Credit(Credit),
        Decision(Decision),
    }

    impl From<Decision> for VarTerms {
        fn from(d: Decision) -> Self {
            Self::Decision(d)
        }
    }

    impl From<Score> for VarTerms {
        fn from(s: Score) -> Self {
            Self::Score(s)
        }
    }

    impl From<Ratio> for VarTerms {
        fn from(r: Ratio) -> Self {
            Self::Ratio(r)
        }
    }

    impl From<Credit> for VarTerms {
        fn from(c: Credit) -> Self {
            Self::Credit(c)
        }
    }

    impl Terms for VarTerms {
        // fn values(&self) -> &'static [(f32, f32)] {
        //     match self {
        //         Self::Score(Score::High) => &[(175., 0.), (180., 0.2), (185., 0.7), (190., 1.)],
        //         Self::Score(Score::Low) => &[
        //             (155., 1.),
        //             (160., 0.8),
        //             (165., 0.5),
        //             (170., 0.2),
        //             (175., 0.),
        //         ],
        //         Self::Ratio(Ratio::Good) => &[(0.3, 1.), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)],
        //         Self::Ratio(Ratio::Bad) => &[(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)],
        //         Self::Credit(Credit::Good) => &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
        //         Self::Credit(Credit::Bad) => &[(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
        //         Self::Decision(Decision::Approve) => &[(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
        //         Self::Decision(Decision::Reject) => &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
        //     }
        // }

        fn build() -> HashMap<Self, Vec<(f32, f32)>> {
            let mut map = HashMap::with_capacity(8);

            map.insert(
                Self::Score(Score::High),
                vec![(175., 0.), (180., 0.2), (185., 0.7), (190., 1.)],
            );
            map.insert(
                Self::Score(Score::Low),
                vec![
                    (155., 1.),
                    (160., 0.8),
                    (165., 0.5),
                    (170., 0.2),
                    (175., 0.),
                ],
            );
            map.insert(
                Self::Ratio(Ratio::Good),
                vec![(0.3, 1.), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)],
            );
            map.insert(
                Self::Ratio(Ratio::Bad),
                vec![(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)],
            );
            map.insert(
                Self::Credit(Credit::Good),
                vec![(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
            );
            map.insert(
                Self::Credit(Credit::Bad),
                vec![(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
            );
            map.insert(
                Self::Decision(Decision::Approve),
                vec![(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)],
            );
            map.insert(
                Self::Decision(Decision::Reject),
                vec![(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
            );
            map
        }
    }

    let mut vars = Variables::<VarTerms>::new();
    let score = vars.add::<Score>(150. ..=200., ZeroOne(1.));
    let ratio = vars.add::<Ratio>(0.1..=1., ZeroOne(1.));
    let credit = vars.add::<Credit>(0. ..=10., ZeroOne(1.));
    // let decision = vars.add::<Decision>(0. ..=10., ZeroOne(1.));
    let mut rules = Rules::new();

    rules.add(
        score
            .is(Score::High)
            .and2(ratio.is(Ratio::Good), credit.is(Credit::Good)),
        Decision::Approve,
    );
    rules.add(
        score
            .is(Score::Low)
            .and(ratio.is(Ratio::Bad).or(credit.is(Credit::Bad))),
        Decision::Reject,
    );

    let mut inputs = Inputs::new();

    inputs.add(score, 190.);
    inputs.add(ratio, 0.39);
    inputs.add(credit, 1.5);

    let model = DecompInference::new(
        AndOp::Min,
        OrOp::Max,
        CompositionOp::MaxMin,
        ImplicationOp::Rc,
        ProductionLink::Max,
        DefuzzificationOp::Cog,
    );

    model.eval(&mut vars, &rules, &inputs);
}
