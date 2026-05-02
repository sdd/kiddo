use std::mem::MaybeUninit;

use crate::{Axis, StemStrategy};

const INLINE_QUERY_STACK_CAPACITY: usize = 50;

pub trait ScalarStackContext<A, S>: Sized {
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self;
    fn into_parts(self) -> (S, A, A);

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, _restore_dim: usize, old_off: A, rd: A) -> Self {
        Self::from_parts(stem_state, old_off, rd)
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        let (stem_state, old_off, rd) = self.into_parts();
        (stem_state, None, old_off, rd)
    }
}

/// Trait for query stack types to enable generic backtracking implementations
pub trait StackTrait<A, SS: StemStrategy> {
    fn push(&mut self, item: SS::StackContext<A>);
    fn pop(&mut self) -> Option<SS::StackContext<A>>;
    fn clear(&mut self);
}

#[derive(Debug)]
pub struct QueryStack<A, SS: StemStrategy> {
    stack: [MaybeUninit<SS::StackContext<A>>; INLINE_QUERY_STACK_CAPACITY],
    spill: Vec<SS::StackContext<A>>,
    len: usize,
}

impl<A, SS: StemStrategy> Default for QueryStack<A, SS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct QueryStackContext<A, S> {
    pub stem_state: S,
    pub restore_dim: Option<usize>,
    pub old_off: A,
    pub rd: A,
}

impl<A, SS: StemStrategy> StackTrait<A, SS> for QueryStack<A, SS> {
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        if self.len < INLINE_QUERY_STACK_CAPACITY {
            unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        } else {
            self.spill.push(item);
        }
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            if self.len >= INLINE_QUERY_STACK_CAPACITY {
                Some(self.spill.pop().expect("query stack spill underflow"))
            } else {
                Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
            }
        }
    }

    #[inline]
    fn clear(&mut self) {
        if self.len > INLINE_QUERY_STACK_CAPACITY {
            self.spill.clear();
            self.len = INLINE_QUERY_STACK_CAPACITY;
        }

        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<A, SS: StemStrategy> QueryStack<A, SS> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_QUERY_STACK_CAPACITY],
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

impl<A, SS: StemStrategy> Drop for QueryStack<A, SS> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A: Axis<Coord = A>, S> QueryStackContext<A, S> {
    pub fn new(stem_state: S) -> Self {
        Self {
            stem_state,
            restore_dim: None,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }
}

impl<A, S> QueryStackContext<A, S> {
    pub fn into_parts(self) -> (S, A, A) {
        (self.stem_state, self.old_off, self.rd)
    }
}

impl<A, S> ScalarStackContext<A, S> for QueryStackContext<A, S> {
    #[inline(always)]
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim: None,
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, restore_dim: usize, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim: Some(restore_dim),
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        QueryStackContext::into_parts(self)
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        (self.stem_state, self.restore_dim, self.old_off, self.rd)
    }
}
