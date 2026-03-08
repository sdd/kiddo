use crate::kd_tree::query_stack::StackTrait;
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
        old_off: A,
        rd: A,
    },
    Block {
        siblings: [SS; 8],
        rd_values: [A; 8],
        new_off_values: [A; 8], // Per-sibling offset values (e.g., interval distances)
        sibling_mask: u8,
        dim: usize,
        old_off: A,
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
        new_off_values: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
    ) -> Self {
        Self::Block {
            siblings,
            rd_values,
            new_off_values,
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
