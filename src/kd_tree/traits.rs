pub trait QueryContext<A, O, const K: usize> {
    fn query(&self) -> &[A; K];
    fn max_dist(&self) -> O;
}
