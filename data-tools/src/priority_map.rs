use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::ops::{Add, Sub};

pub struct PriorityMap<K, V> 
where 
    K: Ord + Hash + Clone,
    V: Ord + Clone,
{
    map: HashMap<K, V>,
    set: BTreeSet<(V, K)>,
}

impl<K, V> PriorityMap<K, V> 
where 
    K: Ord + Hash + Clone,
    V: Ord + Clone + Add<Output = V> + Sub<Output = V> + Default,
{
    pub fn new() -> Self {
        PriorityMap {
            map: HashMap::new(),
            set: BTreeSet::new(),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|value| {
            self.set.remove(&(value.clone(), key.clone()));
            value
        })
    }

    pub fn adjust(&mut self, key: K, delta: V) {
        let new_value = if let Some(value) = self.map.get(&key) {
            self.set.remove(&(value.clone(), key.clone()));
            value.clone() + delta
        } else {
            delta
        };

        if new_value == V::default() {
            self.map.remove(&key);
        } else {
            self.map.insert(key.clone(), new_value.clone());
            self.set.insert((new_value, key));
        }
    }

    pub fn get_max(&self) -> Option<&K> {
        self.set.iter().next_back().map(|(_, k)| k)
    }
}