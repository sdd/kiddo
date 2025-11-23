pub trait QueryContext<A, const K: usize> {
    fn query(&self) -> &[A; K];
}

pub trait ResultContext<A> {
    fn max_dist(&self) -> A;
    fn update_bound(&mut self, _new_bound: A) {}
}
