use crate::traits_unified_2::AxisUnified;

#[derive(Debug, Default)]
pub(crate) struct QueryStack<A, SS> {
    stack: Vec<QueryStackContext<A, SS>>,
}

#[derive(Debug)]
pub(crate) struct QueryStackContext<A, SS> {
    pub stem_strat: SS,
    pub dim: usize,
    pub old_off: A,
    pub rd: A,
}

impl<A, SS> QueryStack<A, SS> {
    #[inline]
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    #[inline]
    pub fn push(&mut self, item: QueryStackContext<A, SS>) {
        self.stack.push(item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<QueryStackContext<A, SS>> {
        self.stack.pop()
    }
}

impl<A: AxisUnified<Coord = A>, SS> QueryStackContext<A, SS> {
    pub fn new(stem_strat: SS) -> Self {
        Self {
            stem_strat,
            dim: 0,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn into_parts(self) -> (SS, usize, A, A) {
        (self.stem_strat, self.dim, self.old_off, self.rd)
    }
}
