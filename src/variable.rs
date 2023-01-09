use std::collections::HashMap;
use std::marker::PhantomData;
use std::{hash::Hash, ops::RangeInclusive};

use fixed_map::key::Key as FixedKey;
use slotmap::{new_key_type, SlotMap};

use crate::linspace::Linspace;
use crate::math::interp;
use crate::terms::Terms;

new_key_type! {
    /// A variable key
    pub struct VariableKey;
}

pub struct Variable<I>(pub(crate) VariableKey, PhantomData<I>);

impl<I> Clone for Variable<I> {
    fn clone(&self) -> Self {
        Variable(self.0, PhantomData)
    }
}

impl<I> Copy for Variable<I> {}

#[derive(Default)]
pub struct Variables<T>(pub(crate) SlotMap<VariableKey, VariableContraints<T>>);

impl<T: Eq + Hash> Variables<T> {
    pub fn new() -> Self {
        Self(SlotMap::with_key())
    }

    /// If the step value is not provided, it defaults to 0.1
    pub fn add<I: Into<T> + FixedKey + 'static>(
        &mut self,
        universe_range: RangeInclusive<f64>,
        terms: Terms<I>,
        step: Option<f64>,
    ) -> Variable<I> {
        let start_term_coords = terms.0.iter().map(|(k, v)| (k.into(), *v));
        let key = self.0.insert(VariableContraints::new(
            universe_range,
            start_term_coords,
            terms.0.len(),
            step.unwrap_or(0.1),
        ));

        Variable(key, PhantomData)
    }
}

pub(crate) struct VariableContraints<T> {
    pub(crate) universe: Vec<f64>,
    pub(crate) min_u: f64,
    pub(crate) max_u: f64,
    pub(crate) terms: HashMap<T, Vec<f64>>,
}

impl<T: Eq + Hash> VariableContraints<T> {
    fn new<'t>(
        universe_range: RangeInclusive<f64>,
        start_term_coords: impl IntoIterator<Item = (T, &'t [(f64, f64)])>,
        n_terms: usize,
        step: f64,
    ) -> Self {
        let min_u = *universe_range.start();
        let max_u = *universe_range.end();
        // floor is closest approx to what python does for int() conversion. But at least one edgecase exists
        // where the decimals are really long: int(4.999999999999999999) == 5
        let num = ((max_u - min_u) / step).floor() as usize + 1;
        let universe = Linspace::new(min_u, max_u, num).collect();
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
                this.add_points_to_universe(xp);
                this.terms
                    .insert(term, interp(this.universe.iter().copied(), membership.iter().copied()));
            }
        }

        this
    }

    // TODO: Try and make VariableContraints immutable?
    pub(crate) fn add_points_to_universe(&mut self, points: impl IntoIterator<Item = f64>) {
        // Adds new points to the universe
        let iter = points.into_iter().map(|p| p.clamp(self.min_u, self.max_u));
        let mut universe: Vec<_> = self.universe.iter().copied().chain(iter).collect();

        universe.sort_unstable_by(|a, b| a.partial_cmp(b).expect("not to find unsortable floats"));
        universe.dedup();

        // Expand existent membership functions with the new points
        for term_values in self.terms.values_mut() {
            let new_values = interp(
                universe.iter().copied(),
                self.universe.iter().copied().zip(term_values.iter().copied()),
            );

            *term_values = new_values;
        }

        // Update the universe with the new points
        self.universe = universe;
    }

    #[allow(clippy::let_and_return)]
    pub(crate) fn get_modified_membership(&self, term: &T, _modifiers: &[()]) -> &[f64] {
        let membership = &self.terms[term];

        // TODO: Apply modifiers

        membership
    }
}
