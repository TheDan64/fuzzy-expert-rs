use std::collections::HashMap;
use std::hash::Hash;

use num::Float;

use crate::dsl::Expr;
use crate::inputs::Inputs;
use crate::math::{meshgrid, Axis, CollectMatrix, Matrix};
use crate::ops::*;
use crate::outputs::Outputs;
use crate::rules::Rules;
use crate::variable::{VariableKey, Variables};

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
    pub fn eval<T: Eq + Hash>(&self, vars: &mut Variables<T>, rules: &Rules<T>, inputs: &Inputs) -> Outputs {
        // Convert Inputs to facts
        // Converts input values to FIS facts
        let mut fact_value = HashMap::with_capacity(inputs.0.len());
        let mut fact_cf = HashMap::with_capacity(inputs.0.len());

        for (key, input_value) in &inputs.0 {
            // TODO: Fuzzy inputs
            fact_value.insert(*key, *input_value);
            fact_cf.insert(*key, 1.);
        }

        // #1 slowest section
        // Fuzzificate Facts
        // Convert crisp facts to membership functions
        // let mut fact_types = HashMap::with_capacity(fact_value.len());
        let mut fact_values = HashMap::with_capacity(inputs.0.len());

        for (key, fact_value) in fact_value {
            // py: (float, int?)
            // This will eventually be an enum match
            if true {
                // Fuzzificate Crisp Fact
                // TODO: Get rid of the mutability?
                let var = &mut vars.0[key];

                var.add_points_to_universe(Some(fact_value));
                fact_values.insert(
                    key,
                    var.universe
                        .iter()
                        .map(|u| if *u == fact_value { 1.0f64 } else { 0. })
                        .collect::<Vec<_>>(),
                );
                // py: list
            } else {
                // TODO: Fuzzificate fuzzy fact
                unimplemented!();
            }
        }

        // Compute Modified Premise Memberships
        let mut modified_premise_memberships = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            let premise = &rule.premise;

            for (var_key, term, modifiers) in premise.propositions() {
                let membership = vars.0[var_key].get_modified_membership(term, modifiers);
                modified_premise_memberships.insert((i, var_key), membership);
            }
        }

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
            rules.0.len() * modified_premise_memberships.len() * modified_consequence_memberships.len(),
        );

        for (i, premise_name) in modified_premise_memberships.keys() {
            for (j, consequence_name) in modified_consequence_memberships.keys() {
                // TODO: It'd be great if we didn't have to iterate through all other rules' consequences
                // Maybe want to turn modified_consequence_memberships into Map<RuleId, Vec<VariableKey>>?
                if *i != *j {
                    continue;
                }

                let premise_membership = &modified_premise_memberships[&(*i, *premise_name)];
                let consequence_membership = &modified_consequence_memberships[&(*i, *consequence_name)];
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

        // #2 slowest section
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
            for (j, consequence_name) in modified_consequence_memberships.keys() {
                if i != *j {
                    continue;
                }

                // Originally tried to write this without collecting vecs at each layer, but
                // recursive iterators are more or less impossible to write... Even with dyn trait.
                // Not sure how to make this more performant
                fn combine<T, F: Float>(
                    expr: &Expr<T>,
                    fuzzy_compositions: &HashMap<(usize, VariableKey, VariableKey), Vec<F>>,
                    consequence_name: VariableKey,
                    and_op: AndOp,
                    or_op: OrOp,
                    rule_id: usize,
                ) -> Vec<F> {
                    match expr {
                        Expr::Is(var_key, _) => fuzzy_compositions[&(rule_id, *var_key, consequence_name)].clone(),
                        Expr::And(expr, expr2) => {
                            let left = combine(expr, fuzzy_compositions, consequence_name, and_op, or_op, rule_id);
                            let right = combine(expr2, fuzzy_compositions, consequence_name, and_op, or_op, rule_id);

                            and_op.call(left, right).into_iter().collect()
                        },
                        Expr::Or(expr, expr2) => {
                            let left = combine(expr, fuzzy_compositions, consequence_name, and_op, or_op, rule_id);
                            let right = combine(expr2, fuzzy_compositions, consequence_name, and_op, or_op, rule_id);

                            or_op.call(left, right).into_iter().collect()
                        },
                    }
                }

                let combined_composition = combine(
                    &rule.premise,
                    &fuzzy_compositions,
                    *consequence_name,
                    self.and_op,
                    self.or_op,
                    i,
                );

                combined_compositions.insert((i, *consequence_name), combined_composition);
            }
        }

        // Compute Rule Inferred CF
        let mut inferred_cf = HashMap::new();

        for (i, rule) in rules.0.iter().enumerate() {
            fn calc_cf<T>(expr: &Expr<T>, fact_cf: &HashMap<VariableKey, f64>) -> f64 {
                match expr {
                    Expr::Is(var_key, _) => fact_cf[var_key],
                    Expr::And(expr, expr2) => {
                        let left = calc_cf(expr, fact_cf);
                        let right = calc_cf(expr2, fact_cf);

                        f64::min(left, right)
                    },
                    Expr::Or(expr, expr2) => {
                        let left = calc_cf(expr, fact_cf);
                        let right = calc_cf(expr2, fact_cf);

                        f64::max(left, right)
                    },
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
                //     let min = universe.iter().copied().reduce(f64::min).unwrap();
                //     let max = universe.iter().copied().reduce(f64::max).unwrap();
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
                    },
                    [..] => {
                        agg = self
                            .prod_link
                            .call(agg, window[1].iter().copied())
                            .into_iter()
                            .collect();
                    },
                }
            }

            aggregated_memberships.insert(var_key, agg);
        }

        let mut final_inferred_cf = 0.;

        // Aggregate Production CF
        for (i, _rule) in rules.0.iter().enumerate() {
            final_inferred_cf = f64::max(final_inferred_cf, inferred_cf[&i]);
        }

        // Defuzzificate
        let mut defuzzificated_inferred_memberships = HashMap::new();

        for (var_key, aggregated_membership) in aggregated_memberships {
            let var = &vars.0[var_key];

            if aggregated_membership.iter().copied().sum::<f64>() == 0. {
                let mean = var.universe.iter().copied().sum::<f64>() / var.universe.len() as f64;

                defuzzificated_inferred_memberships.insert(var_key, mean);
            } else {
                let defuzzed = self.defuzz_op.call(&var.universe, &aggregated_membership);

                defuzzificated_inferred_memberships.insert(var_key, defuzzed);
            }
        }

        Outputs::new(defuzzificated_inferred_memberships, final_inferred_cf)
    }
}

#[test]
fn test_bank_loan() {
    use crate::terms::Terms;
    use fixed_map::Key;

    #[derive(Clone, Copy, Debug, Eq, Hash, Key, Ord, PartialEq, PartialOrd)]
    enum Score {
        High,
        Low,
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Key, Ord, PartialEq, PartialOrd)]
    enum Ratio {
        Good,
        Bad,
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Key, Ord, PartialEq, PartialOrd)]
    enum Credit {
        Good,
        Bad,
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Key, Ord, PartialEq, PartialOrd)]
    enum Decision {
        Approve,
        Reject,
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

    // TODO: The above lines should all be compressed into a macro_rules macro

    let mut score_terms = Terms::new();
    let mut ratio_terms = Terms::new();
    let mut credit_terms = Terms::new();
    let mut decision_terms = Terms::new();

    score_terms.insert(Score::High, &[(175.0, 0.0), (180., 0.2), (185., 0.7), (190., 1.)]);
    score_terms.insert(
        Score::Low,
        &[(155.0, 1.0), (160., 0.8), (165., 0.5), (170., 0.2), (175., 0.)],
    );
    ratio_terms.insert(Ratio::Good, &[(0.3, 1.0), (0.4, 0.7), (0.41, 0.3), (0.42, 0.)]);
    ratio_terms.insert(Ratio::Bad, &[(0.44, 0.), (0.45, 0.3), (0.5, 0.7), (0.7, 1.)]);
    credit_terms.insert(Credit::Good, &[(2.0, 1.0), (3., 0.7), (4., 0.3), (5., 0.)]);
    credit_terms.insert(Credit::Bad, &[(5., 0.), (6., 0.3), (7., 0.7), (8., 1.)]);
    decision_terms.insert(Decision::Approve, &[(5.0, 0.0), (6., 0.3), (7., 0.7), (8., 1.)]);
    decision_terms.insert(Decision::Reject, &[(2., 1.), (3., 0.7), (4., 0.3), (5., 0.)]);

    let mut vars = Variables::<VarTerms>::new();
    let score = vars.add(150. ..=200., score_terms);
    let ratio = vars.add(0.1..=1., ratio_terms);
    let credit = vars.add(0. ..=10., credit_terms);
    let decision = vars.add(0. ..=10., decision_terms);
    let mut rules = Rules::new();

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

    let outputs = model.eval(&mut vars, &rules, &inputs);

    assert_eq!(outputs.get_inferred_membership(decision), Some(8.010492631084489));
    assert_eq!(outputs.inferred_cf(), 1.);
}
