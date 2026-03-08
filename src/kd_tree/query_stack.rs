use crate::traits_unified_2::AxisUnified;
use crate::StemStrategy;
use std::mem::{ManuallyDrop, MaybeUninit};

const INLINE_QUERY_STACK_CAPACITY: usize = 50;

/// Trait for query stack types to enable generic backtracking implementations
pub trait StackTrait<A, SS: StemStrategy> {
    fn push(&mut self, item: SS::StackContext<A>);
    fn pop(&mut self) -> Option<SS::StackContext<A>>;
    fn clear(&mut self);
}

#[derive(Debug)]
pub struct QueryStack<A, SS: StemStrategy> {
    stack: [MaybeUninit<SS::StackContext<A>>; INLINE_QUERY_STACK_CAPACITY],
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
    // pub dim: usize,
    pub old_off: A,
    pub rd: A,
}

impl<A, SS: StemStrategy> StackTrait<A, SS> for QueryStack<A, SS> {
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        debug_assert!(self.len < INLINE_QUERY_STACK_CAPACITY);
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

impl<A, SS: StemStrategy> QueryStack<A, SS> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_QUERY_STACK_CAPACITY],
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

impl<A: AxisUnified<Coord = A>, S> QueryStackContext<A, S> {
    pub fn new(stem_state: S) -> Self {
        Self {
            stem_state,
            // dim: 0,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn into_parts(self) -> (S, /*usize,*/ A, A) {
        (self.stem_state, /*self.dim,*/ self.old_off, self.rd)
    }
}

#[inline(always)]
pub(crate) fn scalar_ctx_from_parts<A, SS>(
    stem_state: SS::DeferredState,
    old_off: A,
    rd: A,
) -> SS::StackContext<A>
where
    A: AxisUnified<Coord = A>,
    SS: StemStrategy,
{
    debug_assert_eq!(SS::BLOCK_SIZE, 1);
    debug_assert_eq!(
        std::mem::size_of::<SS::StackContext<A>>(),
        std::mem::size_of::<QueryStackContext<A, SS::DeferredState>>()
    );
    debug_assert_eq!(
        std::mem::align_of::<SS::StackContext<A>>(),
        std::mem::align_of::<QueryStackContext<A, SS::DeferredState>>()
    );

    let ctx = ManuallyDrop::new(QueryStackContext {
        stem_state,
        old_off,
        rd,
    });

    unsafe { std::ptr::read((&*ctx as *const QueryStackContext<A, SS::DeferredState>).cast()) }
}

#[inline(always)]
pub(crate) fn scalar_ctx_into_parts<A, SS>(ctx: SS::StackContext<A>) -> (SS::DeferredState, A, A)
where
    A: AxisUnified<Coord = A>,
    SS: StemStrategy,
{
    debug_assert_eq!(SS::BLOCK_SIZE, 1);
    debug_assert_eq!(
        std::mem::size_of::<SS::StackContext<A>>(),
        std::mem::size_of::<QueryStackContext<A, SS::DeferredState>>()
    );
    debug_assert_eq!(
        std::mem::align_of::<SS::StackContext<A>>(),
        std::mem::align_of::<QueryStackContext<A, SS::DeferredState>>()
    );

    let ctx = ManuallyDrop::new(ctx);
    let scalar_ctx: QueryStackContext<A, SS::DeferredState> =
        unsafe { std::ptr::read((&*ctx as *const SS::StackContext<A>).cast()) };
    scalar_ctx.into_parts()
}
