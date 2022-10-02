use std::cmp::Ordering;
use crate::tuned::u16::d4::kdtree::{A, T};

pub struct HeapElement {
    pub distance: A,
    pub item: T,
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Eq for HeapElement {}

impl PartialEq for HeapElement {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl PartialEq<A> for HeapElement {
    fn eq(&self, other: &A) -> bool {
        self.distance == *other
    }
}

impl From<HeapElement> for (A, T) {
    fn from(elem: HeapElement) -> Self {
        (elem.distance, elem.item)
    }
}
