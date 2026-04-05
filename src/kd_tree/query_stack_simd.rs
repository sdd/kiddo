use crate::kd_tree::query_stack::{ScalarStackContext, StackTrait};
use crate::traits_unified_2::AxisUnified;
use crate::StemStrategy;
use std::mem::MaybeUninit;

const INLINE_SIMD_QUERY_STACK_CAPACITY: usize = 50;

#[derive(Debug)]
pub struct SimdQueryStack<A, SS: StemStrategy> {
    stack: [MaybeUninit<SS::StackContext<A>>; INLINE_SIMD_QUERY_STACK_CAPACITY],
    len: usize,
}

impl<A, SS: StemStrategy> Default for SimdQueryStack<A, SS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum SimdQueryStackContext<A, SS> {
    Single {
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    },
    Block3Pending {
        base: SS,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        pending_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
    DeferredBlock {
        base: SS,
        child_base: u8,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
    Block {
        siblings: [SS; 8],
        rd_values: [A; 8],
        new_off_values: [A; 8], // Per-sibling offset values (e.g., interval distances)
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
}

impl<A, SS: StemStrategy> StackTrait<A, SS> for SimdQueryStack<A, SS> {
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        debug_assert!(self.len < INLINE_SIMD_QUERY_STACK_CAPACITY);
        unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
        }
    }

    #[inline]
    fn clear(&mut self) {
        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<A, SS: StemStrategy> SimdQueryStack<A, SS> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_SIMD_QUERY_STACK_CAPACITY],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: SS::StackContext<A>) {
        <Self as StackTrait<A, SS>>::push(self, item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<SS::StackContext<A>> {
        <Self as StackTrait<A, SS>>::pop(self)
    }
}

impl<A, SS: StemStrategy> Drop for SimdQueryStack<A, SS> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A: AxisUnified<Coord = A>, SS: StemStrategy> SimdQueryStackContext<A, SS> {
    pub fn new_single(stem_strat: SS) -> Self {
        Self::Single {
            dim: stem_strat.dim(),
            lower_bound: A::min_value(),
            upper_bound: A::max_value(),
            stem_strat,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn new_block(
        siblings: [SS; 8],
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::Block {
            siblings,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            sibling_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn new_deferred_block(
        base: SS,
        child_base: u8,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::DeferredBlock {
            base,
            child_base,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            sibling_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn new_block3_pending(
        base: SS,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        pending_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::Block3Pending {
            base,
            rd_values,
            new_off_values,
            pending_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn into_parts(self) -> (SS, A, A) {
        match self {
            Self::Single {
                stem_strat,
                dim: _,
                lower_bound: _,
                upper_bound: _,
                old_off,
                rd,
            } => (stem_strat, old_off, rd),
            Self::Block3Pending { .. } | Self::DeferredBlock { .. } | Self::Block { .. } => {
                panic!("into_parts called on block variant")
            }
        }
    }
}

impl<A, SS, S> ScalarStackContext<A, S> for SimdQueryStackContext<A, SS> {
    #[inline(always)]
    fn from_parts(_stem_state: S, _old_off: A, _rd: A) -> Self {
        unreachable!("SIMD stack contexts do not support scalar stack packing")
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        unreachable!("SIMD stack contexts do not support scalar stack unpacking")
    }
}
