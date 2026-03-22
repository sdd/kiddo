use crate::kd_tree::result_collection::ResultCollection;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};
use crate::{BestNeighbour, NearestNeighbour};
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::BinaryHeap;
use std::mem::MaybeUninit;

use fixed::{
    types::extra::{U0, U16, U8},
    FixedI32, FixedU16,
};

// TODO: chunking
#[allow(unused)]
const CHUNK_SIZE: usize = 32;
pub(crate) const LEAF_SCRATCH_CAPACITY: usize = 1024;

struct LeafScratch<O> {
    acc: [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
    coord_wide: [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
}

impl<O> LeafScratch<O> {
    const fn new() -> Self {
        Self {
            acc: [const { MaybeUninit::uninit() }; LEAF_SCRATCH_CAPACITY],
            coord_wide: [const { MaybeUninit::uninit() }; LEAF_SCRATCH_CAPACITY],
        }
    }
}

#[inline]
pub(crate) fn assert_leaf_scratch_capacity(len: usize) {
    assert!(
        len <= LEAF_SCRATCH_CAPACITY,
        "leaf scratch capacity exceeded: required={} capacity={}",
        len,
        LEAF_SCRATCH_CAPACITY
    );
}

#[inline]
unsafe fn leaf_scratch_slice_mut<O>(
    scratch: &mut [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
    len: usize,
) -> &mut [O] {
    debug_assert!(len <= LEAF_SCRATCH_CAPACITY);
    std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut O, len)
}

#[doc(hidden)]
pub trait TlsLeafScratch: AxisUnified<Coord = Self> + 'static {
    fn with_tls_leaf_scratch<R>(len: usize, f: impl FnOnce(&mut [Self], &mut [Self]) -> R) -> R;

    fn assert_tls_leaf_scratch_capacity(len: usize);
}

macro_rules! impl_tls_leaf_scratch {
    ($tls_name:ident, $t:ty) => {
        thread_local! {
            static $tls_name: UnsafeCell<LeafScratch<$t>> =
                const { UnsafeCell::new(LeafScratch::new()) };
        }

        impl TlsLeafScratch for $t {
            #[inline]
            fn with_tls_leaf_scratch<R>(
                len: usize,
                f: impl FnOnce(&mut [Self], &mut [Self]) -> R,
            ) -> R {
                $tls_name.with(|scratch| {
                    let scratch = unsafe { &mut *scratch.get() };
                    let acc = unsafe { leaf_scratch_slice_mut(&mut scratch.acc, len) };
                    let coord_wide =
                        unsafe { leaf_scratch_slice_mut(&mut scratch.coord_wide, len) };
                    f(acc, coord_wide)
                })
            }

            #[inline]
            fn assert_tls_leaf_scratch_capacity(len: usize) {
                assert_leaf_scratch_capacity(len);
            }
        }
    };
}

impl_tls_leaf_scratch!(LEAF_SCRATCH_F32, f32);
impl_tls_leaf_scratch!(LEAF_SCRATCH_F64, f64);
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_I32_U16, FixedI32<U16>);
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_I32_U0, FixedI32<U0>);
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_U16_U8, FixedU16<U8>);

#[cfg(feature = "f16")]
impl_tls_leaf_scratch!(LEAF_SCRATCH_F16, half::f16);

/// A view into a leaf node's data.
///
/// Provides a unified interface for accessing leaf data regardless of the underlying
/// storage strategy.
#[derive(Debug)]
pub struct LeafView<'a, AX, T, const K: usize, const B: usize> {
    points: [&'a [AX]; K],
    items: &'a [T],
}

impl<'a, AX: AxisUnified<Coord = AX>, T: Basics, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    pub(crate) fn new(points: [&'a [AX]; K], items: &'a [T]) -> Self {
        Self { points, items }
    }

    #[allow(dead_code)]
    pub(crate) fn into_parts(self) -> ([&'a [AX]; K], &'a [T]) {
        (self.points, self.items)
    }

    #[inline(always)]
    pub(crate) fn points(&self) -> [&'a [AX]; K] {
        self.points
    }

    #[inline(always)]
    pub(crate) fn items(&self) -> &'a [T] {
        self.items
    }

    #[allow(dead_code)]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn len(&self) -> usize {
        unsafe { self.points.get_unchecked(0).len() }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn nearest_one<D>(
        &self,
        query: &[AX; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        self.nearest_one_with_query_wide::<D>(&query_wide, best_dist, best_item);
    }

    #[inline(always)]
    pub(crate) fn nearest_one_with_query_wide<D>(
        &self,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        if crate::kd_tree::leaf_view_chunked::try_nearest_one_with_query_wide::<AX, T, D, K, B>(
            self, query_wide, best_dist, best_item,
        ) {
            return;
        }

        let n = self.len();
        D::Output::assert_tls_leaf_scratch_capacity(n);

        D::Output::with_tls_leaf_scratch(n, |acc, coord_wide| {
            if n == 0 {
                return;
            }

            let points_dim = unsafe { *self.points.get_unchecked(0) };
            let q = unsafe { *query_wide.get_unchecked(0) };
            let widened_axis =
                if let Some(identity_axis) = try_identity_widen_axis::<AX, D::Output>(points_dim) {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

            let mut idx = 0;
            while idx < n {
                unsafe {
                    *acc.get_unchecked_mut(idx) = D::dist1(*widened_axis.get_unchecked(idx), q);
                }
                idx += 1;
            }

            for dim in 1..K {
                let points_dim = unsafe { *self.points.get_unchecked(dim) };
                let q = unsafe { *query_wide.get_unchecked(dim) };
                let widened_axis = if let Some(identity_axis) =
                    try_identity_widen_axis::<AX, D::Output>(points_dim)
                {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

                let mut idx = 0;
                while idx < n {
                    unsafe {
                        *acc.get_unchecked_mut(idx) +=
                            D::dist1(*widened_axis.get_unchecked(idx), q);
                    }
                    idx += 1;
                }
            }

            let mut leaf_best_idx = 0usize;
            let mut leaf_best_dist = unsafe { *acc.get_unchecked(0) };

            let mut idx = 1usize;
            while idx < n {
                let dist = unsafe { *acc.get_unchecked(idx) };
                if dist < leaf_best_dist {
                    leaf_best_dist = dist;
                    leaf_best_idx = idx;
                }
                idx += 1;
            }

            if leaf_best_dist < *best_dist {
                *best_dist = leaf_best_dist;
                *best_item = unsafe { *self.items.get_unchecked(leaf_best_idx) };
            }
        });
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn nearest_n_within<D, R>(
        &self,
        query: &[AX; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut R,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        R: ResultCollection<D::Output, T>,
        AX: 'static,
    {
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        self.nearest_n_within_with_query_wide::<D, R>(&query_wide, dist, results);
    }

    #[inline]
    pub(crate) fn nearest_n_within_with_query_wide<D, R>(
        &self,
        query_wide: &[D::Output; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut R,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        R: ResultCollection<D::Output, T>,
        AX: 'static,
    {
        self.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
            Self::update_nearest_dists(dists, self.items, dist, results);
        });
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn with_dists_for_slice_wide<D, R>(
        &self,
        query_wide: &[D::Output; K],
        f: impl FnOnce(&[D::Output]) -> R,
    ) -> R
    where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        let n = self.len();
        D::Output::assert_tls_leaf_scratch_capacity(n);

        D::Output::with_tls_leaf_scratch(n, |acc, coord_wide| {
            acc.fill(D::Output::zero());

            for dim in 0..K {
                let points_dim = unsafe { *self.points.get_unchecked(dim) };
                let q = unsafe { *query_wide.get_unchecked(dim) };
                let widened_axis = if let Some(identity_axis) =
                    try_identity_widen_axis::<AX, D::Output>(points_dim)
                {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

                for (dst, &coord) in acc.iter_mut().zip(widened_axis.iter()) {
                    *dst += D::dist1(coord, q);
                }
            }

            f(acc)
        })
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dist<O>(
        dists: &[O],
        items: &[T],
        best_dist: &mut O,
        best_item: &mut T,
    ) where
        O: AxisUnified<Coord = O>,
    {
        let (leaf_best_item, leaf_best_dist) = dists
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if *leaf_best_dist < *best_dist {
            *best_dist = *leaf_best_dist;
            *best_item = items[leaf_best_item];
        }
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dists<O, R>(dists: &[O], items: &[T], dist: O, results: &mut R)
    where
        O: AxisUnified<Coord = O>,
        R: ResultCollection<O, T>,
    {
        dists.iter().zip(items).for_each(|(&d, &i)| {
            if d <= dist {
                results.add(NearestNeighbour {
                    distance: d,
                    item: i,
                });
            }
        })
    }
}

#[inline(always)]
pub(crate) fn try_identity_widen_axis<AX: 'static, O: 'static>(axis: &[AX]) -> Option<&[O]> {
    let same_type = TypeId::of::<AX>() == TypeId::of::<O>();
    let float_identity =
        TypeId::of::<AX>() == TypeId::of::<f32>() || TypeId::of::<AX>() == TypeId::of::<f64>();

    if same_type && float_identity {
        Some(unsafe { std::slice::from_raw_parts(axis.as_ptr() as *const O, axis.len()) })
    } else {
        None
    }
}

impl<'a, AX: AxisUnified<Coord = AX>, T: Basics + Ord, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn best_n_within<D>(
        &self,
        query: &[AX; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut BinaryHeap<BestNeighbour<<D as DistanceMetricUnified<AX, K>>::Output, T>>,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        self.best_n_within_with_query_wide::<D>(&query_wide, dist, results);
    }

    #[inline]
    pub(crate) fn best_n_within_with_query_wide<D>(
        &self,
        query_wide: &[D::Output; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut BinaryHeap<BestNeighbour<<D as DistanceMetricUnified<AX, K>>::Output, T>>,
    ) where
        D: DistanceMetricUnified<AX, K>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        self.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
            Self::update_best_dists(dists, self.items, dist, results.capacity(), results);
        });
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_best_dists<O>(
        dists: &[O],
        items: &[T],
        dist: O,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<O, T>>,
    ) where
        O: AxisUnified<Coord = O>,
    {
        dists.iter().zip(items).for_each(|(&d, &item)| {
            if d <= dist {
                if results.len() < max_qty {
                    results.push(BestNeighbour { distance: d, item });
                } else {
                    let mut top = results.peek_mut().unwrap();
                    if item < top.item {
                        top.item = item;
                        top.distance = d;
                    }
                }
            }
        })
    }
}
