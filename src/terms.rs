pub use fixed_map::key::Key as Term;
pub use fixed_map::Key as Term;
use fixed_map::Map as FixedMap;

pub struct Terms<'t, K: Term>(pub(crate) FixedMap<K, &'t [(f64, f64)]>);

impl<'t, K: Term> Terms<'t, K> {
    pub fn new() -> Self {
        Self(FixedMap::new())
    }

    pub fn insert(&mut self, key: K, value: &'t [(f64, f64)]) {
        self.0.insert(key, value);
    }
}
