use crate::kd_tree::query_stack::StackTrait;
use crate::traits_unified_2::AxisUnified;

#[derive(Debug)]
pub struct SimdQueryStack<A, SS> {
    stack: Vec<SimdQueryStackContext<A, SS>>,
}

impl<A, SS> Default for SimdQueryStack<A, SS> {
    fn default() -> Self {
        Self { stack: Vec::new() }
    }
}

#[derive(Debug)]
pub enum SimdQueryStackContext<A, SS> {
    Single {
        stem_strat: SS,
        old_off: A,
        rd: A,
    },
    Block {
        siblings: [SS; 8],
        rd_values: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
    },
}

impl<A, SS> StackTrait<A, SS> for SimdQueryStack<A, SS> {
    type Context = SimdQueryStackContext<A, SS>;

    #[inline]
    fn push(&mut self, item: Self::Context) {
        self.stack.push(item);
    }

    #[inline]
    fn pop(&mut self) -> Option<Self::Context> {
        self.stack.pop()
    }
}

impl<A, SS> SimdQueryStack<A, SS> {
    #[inline]
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    #[inline]
    pub fn push(&mut self, item: SimdQueryStackContext<A, SS>) {
        self.stack.push(item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<SimdQueryStackContext<A, SS>> {
        self.stack.pop()
    }
}

impl<A: AxisUnified<Coord = A>, SS> SimdQueryStackContext<A, SS> {
    pub fn new_single(stem_strat: SS) -> Self {
        Self::Single {
            stem_strat,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn new_block(
        siblings: [SS; 8],
        rd_values: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
    ) -> Self {
        Self::Block {
            siblings,
            rd_values,
            sibling_mask,
            dim,
            old_off,
        }
    }

    pub fn into_parts(self) -> (SS, A, A) {
        match self {
            Self::Single {
                stem_strat,
                old_off,
                rd,
            } => (stem_strat, old_off, rd),
            Self::Block { .. } => panic!("into_parts called on Block variant"),
        }
    }
}
