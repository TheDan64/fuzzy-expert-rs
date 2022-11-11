use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{AddAssign, RangeInclusive};
use std::{collections::HashMap, iter::Sum};

use fixed_map::key::Key as FixedKey;
use fixed_map::Map as FixedMap;
use num::Float;
use slotmap::{new_key_type, SlotMap};

mod dsl;
mod linspace;
mod math;

use dsl::Expr;
use linspace::Linspace;
use math::{interp, meshgrid, Axis, CollectMatrix, Matrix};

/// A value between zero and one
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct ZeroOne(f32);

new_key_type! {
    /// A variable key
    pub struct VariableKey;
}

// REVIEW: Is this trait necessary anymore?
pub trait Variants: Sized + Copy {
    fn variants() -> &'static [Self];
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

impl<T: Terms + std::fmt::Debug> Variables<T> {
    pub fn new() -> Self {
        Self(SlotMap::with_key())
    }

    pub fn add<I: Into<T> + FixedKey + Variants + 'static + std::fmt::Debug>(
        &mut self,
        universe_range: RangeInclusive<f32>,
        terms: FixedMap<I, &[(f32, f32)]>,
    ) -> Variable<I> {
        let start_term_coords = terms.iter().map(|(k, v)| (k.into(), *v));
        let key = self.0.insert(VariableContraints::new(
            universe_range,
            start_term_coords,
            terms.len(),
        ));

        Variable(key, PhantomData)
    }
}

struct VariableContraints<T> {
    universe: Vec<f32>,
    min_u: f32,
    max_u: f32,
    terms: HashMap<T, Vec<f32>>,
}

impl<T: Terms + std::fmt::Debug> VariableContraints<T> {
    fn new<'t>(
        universe_range: RangeInclusive<f32>,
        start_term_coords: impl IntoIterator<Item = (T, &'t [(f32, f32)])>,
        n_terms: usize,
    ) -> Self {
        // TODO: This is an opt param in the original
        let step = 0.1;
        let min_u = *universe_range.start();
        let max_u = *universe_range.end();
        // floor is closest approx to what python does for int() conversion. But at least one edgecase exists
        // where the decimals are really long: int(4.999999999999999999) == 5
        let num = ((max_u - min_u) / step).floor() as usize + 1;
        let universe = Linspace::new(min_u, max_u, num).collect();
        // dbg!(&universe);
        let mut this = Self {
            universe,
            min_u,
            max_u,
            terms: HashMap::with_capacity(n_terms),
        };

        // Load from tuple?
        if false {
            unimplemented!();
        // Load from list
        } else {
            for (term, membership) in start_term_coords {
                let xp = membership.iter().map(|(xp, _)| *xp);
                // dbg!(term);
                // dbg!(membership.iter().map(|(xp, _)| *xp).collect::<Vec<_>>());
                // dbg!(membership.iter().map(|(_, fp)| *fp).collect::<Vec<_>>());
                // dbg!("Before:", &this.universe);
                this.add_points_to_universe(xp);
                this.terms.insert(
                    term,
                    interp(this.universe.iter().copied(), membership.iter().copied()),
                );
                // dbg!("After:", &this.universe);
                // dbg!(&this.terms[&term]);
            }
        }

        this
    }

    // TODO: Try and make VariableContraints immutable?
    fn add_points_to_universe(&mut self, points: impl IntoIterator<Item = f32>) {
        // Adds new points to the universe
        let iter = points.into_iter().map(|p| p.clamp(self.min_u, self.max_u));
        let mut universe: Vec<_> = self.universe.iter().copied().chain(iter).collect();

        // dbg!(&universe);

        universe.sort_unstable_by(|a, b| a.partial_cmp(b).expect("not to find unsortable floats"));
        universe.dedup();

        // Expand existent membership functions with the new points
        for term_values in self.terms.values_mut() {
            let new_values = interp(
                universe.iter().copied(),
                self.universe
                    .iter()
                    .copied()
                    .zip(term_values.iter().copied()),
            );

            *term_values = new_values;
        }

        // Update the universe with the new points
        self.universe = universe;
    }

    #[allow(clippy::let_and_return)]
    fn get_modified_membership(&self, term: &T, _modifiers: &[()]) -> &[f32] {
        let membership = &self.terms[term];

        // TODO: Apply modifiers

        membership
    }
}

#[derive(Default)]
pub struct Rules<T>(Vec<Rule<T>>);

impl<T: Terms> Rules<T> {
    pub fn new() -> Self {
        Rules(Vec::new())
    }

    pub fn add(&mut self, premise: Expr<T>, consequence: Expr<T>) {
        self.0.push(Rule {
            premise,
            consequence,
            // TODO: CFs should be overridable
            cf: 1.0,
            threshold_cf: 0.,
        });
    }
}

struct Rule<T> {
    // TODO: Rename to condition
    premise: Expr<T>,
    // TODO: Rename to result or output
    consequence: Expr<T>,
    cf: f32,
    threshold_cf: f32,
}

pub trait Terms: Copy + Eq + Hash + Sized {}

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

// TODO: Might map to Fact eventually?
#[derive(Debug)]
pub struct Outputs(HashMap<VariableKey, f32>, f32);

enum Fact {
    Crisp(f32),
    Fuzzy(Vec<(f32, ZeroOne)>),
}

/// And operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
pub enum AndOp {
    Min,
    Prod,
    BoundedProd,
    DrasticProd,
}

impl AndOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        match self {
            Self::Min => ProductionLink::Min.call(u, v),
            Self::Prod => ProductionLink::Prod.call(u, v),
            Self::BoundedProd => ProductionLink::BoundedProd.call(u, v),
            Self::DrasticProd => ProductionLink::DrasticProd.call(u, v),
        }
    }
}

/// Or operator method for combining the compositions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
pub enum OrOp {
    Max,
    ProbOr,
    BoundedSum,
    DrasticSum,
}

impl OrOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        match self {
            Self::Max => ProductionLink::Max.call(u, v),
            Self::ProbOr => ProductionLink::ProbOr.call(u, v),
            Self::BoundedSum => ProductionLink::BoundedSum.call(u, v),
            Self::DrasticSum => ProductionLink::DrasticSum.call(u, v),
        }
    }
}

pub enum CompositionOp {
    MaxMin,
    MaxProd,
}

/// Implication operator method for computing the compisitions of propositions
/// in a fuzzy rule premise.
#[derive(Clone, Copy, Debug)]
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

impl ImplicationOp {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        u.into_iter()
            .zip(v.into_iter())
            .map(move |(u, v)| match self {
                Self::Ra => F::min(F::one(), F::one() - u + v),
                Self::Rm => F::max(F::min(u, v), F::one() - u),
                Self::Rc => F::min(u, v),
                Self::Rb => F::max(F::one() - u, v),
                Self::Rs => {
                    if u <= v {
                        F::one()
                    } else {
                        F::zero()
                    }
                }
                Self::Rg => {
                    if u <= v {
                        F::one()
                    } else {
                        v
                    }
                }
                Self::Rsg => F::min(
                    Self::Rs
                        .call(Some(u), Some(v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                    Self::Rg
                        .call(Some(F::one() - u), Some(F::one() - v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                ),
                Self::Rgs => F::min(
                    Self::Rg
                        .call(Some(u), Some(v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                    Self::Rs
                        .call(Some(F::one() - u), Some(F::one() - v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                ),
                Self::Rgg => F::min(
                    Self::Rg
                        .call(Some(u), Some(v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                    Self::Rg
                        .call(Some(F::one() - u), Some(F::one() - v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                ),
                Self::Rss => F::min(
                    Self::Rs
                        .call(Some(u), Some(v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                    Self::Rs
                        .call(Some(F::one() - u), Some(F::one() - v))
                        .into_iter()
                        .next()
                        .expect("unreachable"),
                ),
            })
    }
}

/// Method for aggregating the consequences of the fuzzy rules
#[derive(Clone, Copy, Debug)]
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

impl ProductionLink {
    pub fn call<F: Float>(
        self,
        u: impl IntoIterator<Item = F>,
        v: impl IntoIterator<Item = F>,
    ) -> impl IntoIterator<Item = F> {
        u.into_iter()
            .zip(v.into_iter())
            .map(move |(u, v)| match self {
                Self::Max => F::max(u, v),
                Self::ProbOr => u + v - u * v,
                Self::BoundedSum => F::min(F::one(), u + v),
                Self::DrasticSum => {
                    if v == F::zero() {
                        u
                    } else if u == F::zero() {
                        v
                    } else {
                        F::one()
                    }
                }
                Self::Min => F::min(u, v),
                Self::Prod => u * v,
                Self::BoundedProd => F::max(F::zero(), u + v - F::one()),
                Self::DrasticProd => {
                    if v == F::zero() {
                        u
                    } else if u == F::one() {
                        v
                    } else {
                        F::zero()
                    }
                }
            })
    }
}

/// Method for defuzzifcating the resulting membership function.
#[derive(Clone, Copy, Debug)]
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

impl DefuzzificationOp {
    pub fn call<F: Float + Sum + AddAssign>(self, universe: &[F], membership: &[F]) -> F {
        match self {
            Self::Cog => {
                let n_areas = universe.len() - 1;
                let mut areas = Vec::with_capacity(n_areas);
                let mut centroids = Vec::with_capacity(n_areas);
                let two = F::one() + F::one();
                let three = two + F::one();

                for i in 0..n_areas {
                    let base = universe[i + 1] - universe[i];
                    let area_rect = F::min(membership[i], membership[i + 1]) * base;
                    let center_rect = universe[i] + base / two;
                    let (area_tria, center_tri) = if membership[i + 1] == membership[i] {
                        (F::zero(), F::zero())
                    } else if membership[i + 1] > membership[i] {
                        (
                            base * F::abs(membership[i + 1] - membership[i]) / two,
                            universe[i] + two / three * base,
                        )
                    } else {
                        (
                            base * F::abs(membership[i + 1] - membership[i]) / two,
                            universe[i] + F::one() / three * base,
                        )
                    };
                    let area = area_rect + area_tria;
                    let center = if area == F::zero() {
                        F::zero()
                    } else {
                        (area_rect * center_rect + area_tria * center_tri) / (area_rect + area_tria)
                    };

                    areas.push(area);
                    centroids.push(center);
                }

                let den = areas.iter().copied().sum::<F>();
                let num = areas
                    .into_iter()
                    .zip(centroids.into_iter())
                    .map(|(area, cent)| area * cent)
                    .sum::<F>();

                num / den
            }
            Self::Boa => {
                let n_areas = universe.len() - 1;
                let mut areas = Vec::with_capacity(n_areas);

                for i_area in 0..n_areas {
                    let base = universe[i_area + 1] - universe[i_area];
                    let area = (membership[i_area] + membership[i_area + 1]) * base
                        / F::from(2.).expect("unreachable");
                    areas.push(area);
                }

                let total_area = areas.iter().copied().sum::<F>();
                let target = total_area / F::from(2.).expect("unreachable");
                let mut cum_area = F::zero();
                let mut i_area = 0;

                for i in 0..=n_areas {
                    cum_area += areas[i];
                    i_area = i;
                    if cum_area >= target {
                        break;
                    }
                }

                let xp = [universe[i_area], universe[i_area + 1]];
                let fp = [cum_area - areas[i_area], cum_area];

                interp(Some(target), xp.into_iter().zip(fp.into_iter()))
                    .into_iter()
                    .next()
                    .expect("unreachable")
            }
            Self::Mom => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                let (len, sum) = universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .enumerate()
                    .fold((0usize, F::zero()), |(_i, accum), (i, next)| {
                        (i + 1, accum + next)
                    });

                sum / F::from(len).unwrap()
            }
            Self::Lom => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .reduce(F::max)
                    .unwrap()
            }
            Self::Som => {
                let maximum = membership.iter().copied().reduce(F::max).unwrap();
                universe
                    .iter()
                    .copied()
                    .zip(membership.iter().copied())
                    .filter_map(|(u, m)| if m == maximum { Some(u) } else { None })
                    .reduce(F::min)
                    .unwrap()
            }
        }
    }
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

    // TODO: Maybe can make vars and rules immutable if they're just starting state
    // and everything else is calculated here
    pub fn eval<'r, T: Terms + std::fmt::Debug>(
        &self,
        vars: &mut Variables<T>,
        rules: &'r Rules<T>,
        inputs: &Inputs,
    ) -> Outputs {
        // Convert Inputs to facts
        // Converts input values to FIS facts
        let mut fact_value = HashMap::with_capacity(inputs.0.len());
        let mut fact_cf = HashMap::with_capacity(inputs.0.len());

        for (key, input_value) in &inputs.0 {
            // TODO: Fuzzy inputs
            fact_value.insert(*key, *input_value);
            fact_cf.insert(*key, 1.);
        }

        // Fuzzificate Facts
        // Convert crisp facts to membership functions
        // let mut fact_types = HashMap::with_capacity(fact_value.len());
        let mut fact_values = HashMap::with_capacity(inputs.0.len());

        for (key, fact_value) in fact_value {
            // py: (float, int?)
            // This will eventually be an enum match
            if true {
                // Fuzzificate Crisp Fact
                let var = &mut vars.0[key];

                var.add_points_to_universe(Some(fact_value));
                fact_values.insert(
                    key,
                    var.universe
                        .iter()
                        .map(|u| if *u == fact_value { 1.0f32 } else { 0. })
                        .collect::<Vec<_>>(),
                );
                // fact_types.insert(key, "Crisp");
                // py: list
            } else {
                // TODO: Fuzzificate fuzzy fact
                // fact_types.insert(key, "Fuzzy");
                unimplemented!();
            }
        }

        // Compute Modified Premise Memberships
        let mut modified_premise_memberships = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            let premise = &rule.premise;

            for (var_key, term, modifiers) in premise.propositions() {
                let membership = vars.0[var_key].get_modified_membership(term, modifiers);
                // dbg!((term, i, membership));
                modified_premise_memberships.insert((i, var_key), membership);
            }
        }

        // dbg!(&modified_premise_memberships);

        // Compute Modified Consequence Memberships
        let mut modified_consequence_memberships = HashMap::new();

        // TODO: Can probably move the inner loop into the previous section's loop
        for (i, rule) in rules.0.iter().enumerate() {
            let consequence = &rule.consequence;

            for (var_key, term, modifiers) in consequence.propositions() {
                let membership = vars.0[var_key].get_modified_membership(term, modifiers);
                modified_consequence_memberships.insert((i, var_key), membership);
            }
        }

        // Compute Fuzzy Implication
        let mut fuzzy_implications = HashMap::with_capacity(
            rules.0.len()
                * modified_premise_memberships.len()
                * modified_consequence_memberships.len(),
        );

        for (i, premise_name) in modified_premise_memberships.keys() {
            for (j, consequence_name) in modified_consequence_memberships.keys() {
                // TODO: It'd be great if we didn't have to iterate through all other rules' consequences
                // Maybe want to turn modified_consequence_memberships into Map<RuleId, Vec<VariableKey>>?
                if *i != *j {
                    continue;
                }

                let premise_membership = &modified_premise_memberships[&(*i, *premise_name)];
                let consequence_membership =
                    &modified_consequence_memberships[&(*i, *consequence_name)];
                let (v, u) = meshgrid(
                    consequence_membership.iter().copied(),
                    premise_membership.iter().copied(),
                );
                let shape = v.shape();

                fuzzy_implications.insert(
                    (i, *premise_name, *consequence_name),
                    self.imp_op.call(u, v).collect_matrix(shape),
                );
            }
        }

        // Compute Fuzzy Composition
        let mut fuzzy_compositions = HashMap::with_capacity(fuzzy_implications.len());

        for (i, premise_name) in modified_premise_memberships.keys() {
            for (j, consequence_name) in modified_consequence_memberships.keys() {
                if *i != *j {
                    continue;
                }

                let implication = &fuzzy_implications[&(i, *premise_name, *consequence_name)];
                let fact_values = &fact_values[premise_name];
                let n_dim = fact_values.len();
                let fact_value = Matrix::new(fact_values.to_owned(), (n_dim, 1));
                let fact_value = fact_value.tile((1, implication.shape().1));
                let shape = fact_value.shape();

                debug_assert_eq!(shape, implication.shape());

                let composition = match self.comp_op {
                    CompositionOp::MaxMin => ProductionLink::Min
                        .call(fact_value, implication)
                        .collect_matrix(shape)
                        .max(Axis::Column),
                    CompositionOp::MaxProd => unimplemented!("fact_value * implication"), // TODO: Matrix::from_mul(fact_value, implication)?
                };

                fuzzy_compositions.insert((*i, *premise_name, *consequence_name), composition);
            }
        }

        // Combine Antecedents
        let mut combined_compositions = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            for (j, conseqence_name) in modified_consequence_memberships.keys() {
                if i != *j {
                    continue;
                }

                // Originally tried to write this without collecting vecs at each layer, but
                // recursive iterators are more or less impossible to write... Even with dyn trait.
                // Not sure how to make this more performant
                fn combine<T, F: Float>(
                    expr: &Expr<T>,
                    fuzzy_compositions: &HashMap<(usize, VariableKey, VariableKey), Vec<F>>,
                    conseqence_name: VariableKey,
                    and_op: AndOp,
                    or_op: OrOp,
                    rule_id: usize,
                ) -> Vec<F> {
                    match expr {
                        Expr::Is(var_key, _) => {
                            fuzzy_compositions[&(rule_id, *var_key, conseqence_name)].clone()
                        }
                        Expr::And(expr, expr2) => {
                            let left = combine(
                                expr,
                                fuzzy_compositions,
                                conseqence_name,
                                and_op,
                                or_op,
                                rule_id,
                            );
                            let right = combine(
                                expr2,
                                fuzzy_compositions,
                                conseqence_name,
                                and_op,
                                or_op,
                                rule_id,
                            );

                            and_op.call(left, right).into_iter().collect()
                        }
                        Expr::Or(expr, expr2) => {
                            let left = combine(
                                expr,
                                fuzzy_compositions,
                                conseqence_name,
                                and_op,
                                or_op,
                                rule_id,
                            );
                            let right = combine(
                                expr2,
                                fuzzy_compositions,
                                conseqence_name,
                                and_op,
                                or_op,
                                rule_id,
                            );

                            or_op.call(left, right).into_iter().collect()
                        }
                    }
                }

                let combined_composition = combine(
                    &rule.premise,
                    &fuzzy_compositions,
                    *conseqence_name,
                    self.and_op,
                    self.or_op,
                    i,
                );

                combined_compositions.insert((i, *conseqence_name), combined_composition);
            }
        }

        // Compute Rule Inferred CF
        let mut inferred_cf = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            fn calc_cf<T>(expr: &Expr<T>, fact_cf: &HashMap<VariableKey, f32>) -> f32 {
                match expr {
                    Expr::Is(var_key, _) => fact_cf[var_key],
                    Expr::And(expr, expr2) => {
                        let left = calc_cf(expr, fact_cf);
                        let right = calc_cf(expr2, fact_cf);

                        f32::min(left, right)
                    }
                    Expr::Or(expr, expr2) => {
                        let left = calc_cf(expr, fact_cf);
                        let right = calc_cf(expr2, fact_cf);

                        f32::max(left, right)
                    }
                }
            }

            let aggregated_premise_cf = calc_cf(&rule.premise, &fact_cf);

            inferred_cf.insert(i, aggregated_premise_cf * rule.cf);
        }

        // Collect Rule Memberships
        let mut collected_rule_memberships = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            for (j, var_key) in combined_compositions.keys() {
                if i != *j {
                    continue;
                }

                // REVIEW: Is this even necessary?
                // if !collected_rule_memberships.contains_key(var_key) {
                //     let universe = &vars.0[*var_key].universe;
                //     let min = universe.iter().copied().reduce(f32::min).unwrap();
                //     let max = universe.iter().copied().reduce(f32::max).unwrap();
                //     let mut var = VariableContraints::<T>::new(min..=max, std::iter::empty());
                //     var.universe = universe.clone();

                //     collected_rule_memberships.insert(*var_key, var);
                // }

                if inferred_cf[&i] >= rule.threshold_cf {
                    collected_rule_memberships
                        .entry(*var_key)
                        .or_insert_with(Vec::new)
                        .push(&*combined_compositions[&(i, *var_key)]);
                }
            }
        }

        // Aggregate Collected Memberships
        let mut aggregated_memberships = HashMap::new();

        for (var_key, memberships) in collected_rule_memberships {
            let mut agg = Vec::new();

            for window in memberships.windows(2) {
                match &agg[..] {
                    [] => {
                        agg = self
                            .prod_link
                            .call(window[0].iter().copied(), window[1].iter().copied())
                            .into_iter()
                            .collect();
                    }
                    [..] => {
                        agg = self
                            .prod_link
                            .call(agg, window[1].iter().copied())
                            .into_iter()
                            .collect();
                    }
                }
            }

            aggregated_memberships.insert(var_key, agg);
        }

        let mut final_inferred_cf = 0.;

        // Aggregate Production CF
        for (i, _rule) in rules.0.iter().enumerate() {
            final_inferred_cf = f32::max(final_inferred_cf, inferred_cf[&i]);
        }

        // Defuzzificate
        let mut defuzzificated_infered_memberships = HashMap::new();

        for (var_key, aggregated_membership) in aggregated_memberships {
            let var = &vars.0[var_key];

            if aggregated_membership.iter().copied().sum::<f32>() == 0. {
                let mean = var.universe.iter().copied().sum::<f32>() / var.universe.len() as f32;

                defuzzificated_infered_memberships.insert(var_key, mean);
            } else {
                let defuzzed = self.defuzz_op.call(&var.universe, &aggregated_membership);

                defuzzificated_infered_memberships.insert(var_key, defuzzed);
            }
        }

        Outputs(defuzzificated_infered_memberships, final_inferred_cf)
    }
}

#[test]
fn test_bank_loan() {
    use fixed_map::Key as FixedKey;

    #[derive(Clone, Copy, Debug, Eq, FixedKey, Hash, Ord, PartialEq, PartialOrd)]
    enum Score {
        High,
        Low,
    }

    impl Variants for Score {
        fn variants() -> &'static [Self] {
            &[Self::High, Self::Low]
        }
    }

    #[derive(Clone, Copy, Debug, Eq, FixedKey, Hash, Ord, PartialEq, PartialOrd)]
    enum Ratio {
        Good,
        Bad,
    }

    impl Variants for Ratio {
        fn variants() -> &'static [Self] {
            &[Self::Good, Self::Bad]
        }
    }

    #[derive(Clone, Copy, Debug, Eq, FixedKey, Hash, Ord, PartialEq, PartialOrd)]
    enum Credit {
        Good,
        Bad,
    }

    impl Variants for Credit {
        fn variants() -> &'static [Self] {
            &[Self::Good, Self::Bad]
        }
    }

    #[derive(Clone, Copy, Debug, Eq, FixedKey, Hash, Ord, PartialEq, PartialOrd)]
    enum Decision {
        Approve,
        Reject,
    }

    impl Variants for Decision {
        fn variants() -> &'static [Self] {
            &[Self::Approve, Self::Reject]
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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

    impl Terms for VarTerms {}

    // TODO: The above lines should all be compressed into a macro_rules macro

    let mut score_terms = FixedMap::new();
    let mut ratio_terms = FixedMap::new();
    let mut credit_terms = FixedMap::new();
    let mut decision_terms = FixedMap::new();

    score_terms.insert(
        Score::High,
        &[(175.0f32, 0.0f32), (180., 0.2), (185., 0.7), (190., 1.)] as &[_],
    );
    score_terms.insert(
        Score::Low,
        &[
            (155.0f32, 1.0f32),
            (160., 0.8),
            (165., 0.5),
            (170., 0.2),
            (175., 0.),
        ],
    );
    ratio_terms.insert(
        Ratio::Good,
        &[(0.3f32, 1.0f32), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)] as &[_],
    );
    ratio_terms.insert(
        Ratio::Bad,
        &[(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)],
    );
    credit_terms.insert(
        Credit::Good,
        &[(2.0f32, 1.0f32), (3., 0.7), (4., 0.3), (5., 0.)] as &[_],
    );
    credit_terms.insert(Credit::Bad, &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)]);
    decision_terms.insert(
        Decision::Approve,
        &[(5.0f32, 0.0f32), (6., 0.3), (7., 0.7), (8., 1.)] as &[_],
    );
    decision_terms.insert(
        Decision::Reject,
        &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)],
    );

    let mut vars = Variables::<VarTerms>::new();
    let score = vars.add(150. ..=200., score_terms);
    let ratio = vars.add(0.1..=1., ratio_terms);
    let credit = vars.add(0. ..=10., credit_terms);
    let decision = vars.add(0. ..=10., decision_terms);
    let mut rules = Rules::new();

    // dbg!(("Score", score.0));
    // dbg!(("Ratio", ratio.0));
    // dbg!(("Credit", credit.0));

    rules.add(
        score
            .is(Score::High)
            .and2(ratio.is(Ratio::Good), credit.is(Credit::Good)),
        decision.is(Decision::Approve),
    );
    rules.add(
        score
            .is(Score::Low)
            .and(ratio.is(Ratio::Bad))
            .or(credit.is(Credit::Bad)),
        decision.is(Decision::Reject),
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

    let output = model.eval(&mut vars, &rules, &inputs);

    dbg!(&output);
}
