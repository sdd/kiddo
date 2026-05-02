use std::mem::MaybeUninit;

use crate::kd_tree::query_stack::{ScalarStackContext, StackTrait};
use crate::{Axis, StemStrategy};

const DEFAULT_INLINE_SIMD_QUERY_STACK_CAPACITY: usize = 50;
// Block3 exact traversal pushes at most one pending context per 3-level block.
// Supporting trees up to 30 stem levels therefore requires ceil(30 / 3) = 10 entries.
pub(crate) const BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY: usize = 10;

#[derive(Debug)]
pub struct SimdQueryStack<
    A,
    SS: StemStrategy,
    const INLINE_CAPACITY: usize = DEFAULT_INLINE_SIMD_QUERY_STACK_CAPACITY,
> {
    stack: [MaybeUninit<SS::StackContext<A>>; INLINE_CAPACITY],
    spill: Vec<SS::StackContext<A>>,
    len: usize,
}

#[derive(Debug)]
pub struct Block3ExactQueryStack<
    A,
    SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS, K>>,
    const K: usize,
    const INLINE_CAPACITY: usize = BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY,
> {
    stack: [MaybeUninit<Block3SimdQueryStackContext<A, SS, K>>; INLINE_CAPACITY],
    len: usize,
}

pub trait SimdIntervalStackContext<A, SS: StemStrategy>: Sized {
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self;
}

#[derive(Debug)]
pub enum Block3ExactStackContextState<A, SS, const K: usize> {
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
        pending_mask: u8,
        rd: A,
        lower: [A; K],
        upper: [A; K],
    },
}

pub trait Block3ExactStackContext<A, SS: StemStrategy, const K: usize>: Sized {
    fn new_single(stem_strat: SS) -> Self
    where
        A: Axis<Coord = A>;

    fn new_block3_pending_from_state<const K2: usize>(
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: &[A; K2],
        upper: &[A; K2],
    ) -> Self
    where
        A: Axis<Coord = A>;

    fn into_block3_exact_state(self) -> Block3ExactStackContextState<A, SS, K>;
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> Default
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS, K>>,
        const K: usize,
        const INLINE_CAPACITY: usize,
    > Default for Block3ExactQueryStack<A, SS, K, INLINE_CAPACITY>
{
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
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
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

#[derive(Debug)]
pub enum Block3SimdQueryStackContext<A, SS, const K: usize> {
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
        pending_mask: u8,
        rd: A,
        lower: [A; K],
        upper: [A; K],
    },
}

#[inline(always)]
fn copy_state_array<A, const SRC: usize, const DST: usize>(src: &[A; SRC]) -> [A; DST]
where
    A: Axis<Coord = A>,
{
    assert_eq!(
        SRC, DST,
        "Block3 exact stack context dimension mismatch: src={SRC}, dst={DST}"
    );
    let mut dst = [A::zero(); DST];
    dst.copy_from_slice(&src[..DST]);
    dst
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> StackTrait<A, SS>
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        if self.len < INLINE_CAPACITY {
            unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        } else {
            self.spill.push(item);
        }
        self.len += 1;

        #[cfg(feature = "test_utils")]
        crate::test_utils::exact_query_stats::record_simd_stack_len(self.len);
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            if self.len >= INLINE_CAPACITY {
                Some(self.spill.pop().expect("simd query stack spill underflow"))
            } else {
                Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
            }
        }
    }

    #[inline]
    fn clear(&mut self) {
        if self.len > INLINE_CAPACITY {
            self.spill.clear();
            self.len = INLINE_CAPACITY;
        }

        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS, K>>,
        const K: usize,
        const INLINE_CAPACITY: usize,
    > StackTrait<A, SS> for Block3ExactQueryStack<A, SS, K, INLINE_CAPACITY>
{
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        debug_assert!(
            self.len < INLINE_CAPACITY,
            "Block3 exact stack overflow: len={}, capacity={INLINE_CAPACITY}. Increase BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY.",
            self.len
        );
        unsafe { core::hint::assert_unchecked(self.len < INLINE_CAPACITY) };

        unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        self.len += 1;

        #[cfg(feature = "test_utils")]
        crate::test_utils::exact_query_stats::record_simd_stack_len(self.len);
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

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> SimdQueryStack<A, SS, INLINE_CAPACITY> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
            spill: Vec::new(),
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

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS, K>>,
        const K: usize,
        const INLINE_CAPACITY: usize,
    > Block3ExactQueryStack<A, SS, K, INLINE_CAPACITY>
{
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
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

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> Drop
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    fn drop(&mut self) {
        self.clear();
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS, K>>,
        const K: usize,
        const INLINE_CAPACITY: usize,
    > Drop for Block3ExactQueryStack<A, SS, K, INLINE_CAPACITY>
{
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A: Axis<Coord = A>, SS: StemStrategy> SimdQueryStackContext<A, SS> {
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
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
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
            lower_bounds,
            upper_bounds,
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

impl<A, SS> SimdIntervalStackContext<A, SS> for SimdQueryStackContext<A, SS>
where
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self {
        Self::Single {
            stem_strat,
            dim,
            lower_bound,
            upper_bound,
            old_off,
            rd,
        }
    }
}

impl<A: Axis<Coord = A>, SS: StemStrategy, const K: usize> Block3SimdQueryStackContext<A, SS, K> {
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

    pub fn new_block3_pending(
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: &[A; K],
        upper: &[A; K],
    ) -> Self {
        Self::Block3Pending {
            base,
            pending_mask,
            rd,
            lower: *lower,
            upper: *upper,
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
            Self::Block3Pending { .. } => panic!("into_parts called on block variant"),
        }
    }
}

impl<A, SS, const KCTX: usize, const K: usize> Block3ExactStackContext<A, SS, K>
    for Block3SimdQueryStackContext<A, SS, KCTX>
where
    A: Axis<Coord = A>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single(stem_strat: SS) -> Self
    where
        A: Axis<Coord = A>,
    {
        Block3SimdQueryStackContext::new_single(stem_strat)
    }

    #[inline(always)]
    fn new_block3_pending_from_state<const K2: usize>(
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: &[A; K2],
        upper: &[A; K2],
    ) -> Self
    where
        A: Axis<Coord = A>,
    {
        Block3SimdQueryStackContext::new_block3_pending(
            base,
            pending_mask,
            rd,
            &copy_state_array(lower),
            &copy_state_array(upper),
        )
    }

    #[inline(always)]
    fn into_block3_exact_state(self) -> Block3ExactStackContextState<A, SS, K> {
        match self {
            Self::Single {
                stem_strat,
                dim,
                lower_bound,
                upper_bound,
                old_off,
                rd,
            } => Block3ExactStackContextState::Single {
                stem_strat,
                dim,
                lower_bound,
                upper_bound,
                old_off,
                rd,
            },
            Self::Block3Pending {
                base,
                pending_mask,
                rd,
                lower,
                upper,
            } => Block3ExactStackContextState::Block3Pending {
                base,
                pending_mask,
                rd,
                lower: copy_state_array(&lower),
                upper: copy_state_array(&upper),
            },
        }
    }
}

impl<A, SS, const K: usize> SimdIntervalStackContext<A, SS>
    for Block3SimdQueryStackContext<A, SS, K>
where
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self {
        Self::Single {
            stem_strat,
            dim,
            lower_bound,
            upper_bound,
            old_off,
            rd,
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

impl<A, SS, S, const K: usize> ScalarStackContext<A, S> for Block3SimdQueryStackContext<A, SS, K> {
    #[inline(always)]
    fn from_parts(_stem_state: S, _old_off: A, _rd: A) -> Self {
        unreachable!("SIMD stack contexts do not support scalar stack packing")
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        unreachable!("SIMD stack contexts do not support scalar stack unpacking")
    }
}
