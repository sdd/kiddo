//! Immutable Floating point k-d B-tree.
//!
//! [`ImmutableKdBTree`] offers improved memory utilisation, smaller size
//! when serialised, and faster more consistent query performance, when compared to [`crate::mutable::float::kdtree::KdTree`].
//! This comes at the expense of not being able to modify the contents of the tree after its initial
//! construction, and potentially longer construction times.
//! As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
//! values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if used with the
//! [`half`](https://docs.rs/half/latest/half) crate.
//!
//! ## Normal Usage
//! Most of the structs listed in these docs are only relevant when using `rkyv` for zero-copy
//! serialisation. **The main Struct in here, [`ImmutableKdBTree`], is usually what you're looking for.**
//!
//! ## Rkyv Usage
//! This release of Kiddo supports usage of Rkyv 0.8 only. Kiddo v5.x was the last version of Kiddo
//! with support for Rkyv 0.7.x.
//! Rkyv 0.8 support is now gated behind the `rkyv` crate feature (previously in later versions of
//! Kiddo v5, this was behind the `rkyv_08` feature).
//!
//! See the examples folder for examples of serializing and deserializing
//! with rkyv 0.8, using both the full-deserialize, ZC checked, and ZC unchecked approaches, along with
//! the timings of each approach.
//!
//! ### Using both Rkyv and `f16` / `half` support at the same time
//! If you are using Kiddo's `rkyv` feature and want to use `f16`, bear in mind that only
//! [`half`](https://docs.rs/half/latest/half) 2.5.0 onwards support `rkyv` 0.8.
//!
#[cfg(feature = "rkyv_08")]
use crate::immutable::float::rkyv_aligned_vec::EncodeAVec;
use crate::leaf_slice::float::{LeafSlice, LeafSliceFloat, LeafSliceFloatChunk};
use crate::stem_strategies::Donnelly;
use crate::traits::{Axis, Content, StemStrategy};
use aligned_vec::{avec, AVec, ConstAlign, CACHELINE_ALIGN};
use array_init::array_init;
use az::{Az, Cast};
use ordered_float::OrderedFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rkyv_08")]
use std::fmt::Formatter;
use std::ptr::NonNull;
use std::{cmp::PartialEq, fmt::Debug};

/// Immutable floating point k-d B-tree
///
/// Offers less memory utilisation, smaller size vs non-immutable tree
/// when serialised, and faster more consistent query performance. This comes at the
/// expense of not being able to modify the contents of the tree after its initial
/// construction, and longer construction times.
///
/// Compared to non-dynamic `ImmutableKdBTree`, this can handle data like point clouds
/// that may have many occurrences of multiple points have the exact same value on a given axis.
/// This comes at the expense of slower performance. Memory usage should still be very efficient,
/// more so than the standard and non-dynamic immutable tree types.
///
/// As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
/// values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if used with the
/// [`half`](https://docs.rs/half/latest/half) crate.
///
/// A convenient type alias exists for ImmutableKdBTree with some sensible defaults set: [`kiddo::ImmutableKdBTree`](`crate::ImmutableKdBTree`).
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate=rkyv_08, archived=ArchivedR8ImmutableKdBTree, resolver=ImmutableKdBTreeR8Resolver))]
#[derive(Clone, Debug, PartialEq)]
pub struct ImmutableKdBTree<
    A: Copy + Default,
    T: Copy + Default,
    const K: usize,
    const B: usize,
    const B2: usize,
> {
    #[cfg_attr(feature = "rkyv_08", rkyv(with = EncodeAVec<[A, B2]>))]
    pub(crate) stems: AVec<[A; B2]>,

    #[cfg_attr(feature = "serde", serde(with = "crate::custom_serde::array_of_vecs"))]
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) leaf_points: [Vec<A>; K],
    pub(crate) leaf_items: Vec<T>,
    pub(crate) leaf_extents: Vec<(u32, u32)>,
    pub(crate) max_stem_level: i32,
}

#[cfg(feature = "rkyv_08")]
impl<
        A: Copy + Default + rkyv_08::Archive,
        T: Copy + Default + rkyv_08::Archive,
        const K: usize,
        const B: usize,
        const B2: usize,
    > Debug for ArchivedR8ImmutableKdBTree<A, T, K, B, B2>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // just log out the generic types and size
        write!(
            f,
            "ArchivedR8ImmutableKdBTree<{}, {}, {}, {}, {}> with {} items",
            std::any::type_name::<A>(),
            std::any::type_name::<T>(),
            K,
            B,
            B2,
            self.leaf_items.len()
        )
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize, const B2: usize>
    ArchivedR8ImmutableKdBTree<A, T, K, B, B2>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K> + rkyv_08::Archive,
    T: Content + rkyv_08::Archive,
    usize: Cast<T>,
{
    /// Returns the current number of elements stored in the tree
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn size(&self) -> usize {
        self.leaf_items.len()
    }

    /// Returns the number of stem levels in the tree
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn max_stem_level(&self) -> i32 {
        Into::<i32>::into(self.max_stem_level)
    }

    /// Returns a LeafSlice for a given leaf index
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn get_leaf_slice(&self, leaf_idx: usize) -> LeafSlice<'_, A, T, K> {
        let extents = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
        let start = Into::<u32>::into(extents.0) as usize;
        let end = Into::<u32>::into(extents.1) as usize;

        // Artificially extend size to be at least chunk length for faster processing
        // TODO: why does this slow things down?
        // let end = end.max(start + 32).min(self.leaf_items.len() as u32);

        // Safety: For primitive types like f32/f64/u32/u64, Archived<T> has the same
        // memory layout as T so we can safely reinterpret the slice
        LeafSlice::new(
            array_init::array_init(|i| unsafe {
                std::slice::from_raw_parts(
                    self.leaf_points[i].as_ptr().add(start) as *const A,
                    end - start,
                )
            }),
            unsafe {
                std::slice::from_raw_parts(
                    self.leaf_items.as_ptr().add(start) as *const T,
                    end - start,
                )
            },
        )
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, const B2: usize> From<&[[A; K]]>
    for ImmutableKdBTree<A, T, K, B, B2>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdBTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdBTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdBTree;
    /// use kiddo::Eytzinger;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdBTree<f64, u32, Eytzinger<3>, 3, 32> = (&*points).into();
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    fn from(slice: &[[A; K]]) -> Self {
        ImmutableKdBTree::new_from_slice(slice)
    }
}

// prevent clippy complaining that the feature unreliable_select_nth_unstable
// is not defined (I don't want to explicitly define it as if I do then
// passing --all-features in CI will enable it, which I don't want to do)
#[allow(unexpected_cfgs)]
impl<A, T, const K: usize, const B: usize, const B2: usize> ImmutableKdBTree<A, T, K, B, B2>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdBTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdBTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdBTree;
    /// use kiddo::Eytzinger;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdBTree<f64, u32, Eytzinger<3>, 3, 32> = ImmutableKdBTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn new_from_slice(source: &[[A; K]]) -> Self
    where
        usize: Cast<T>,
    {
        let item_count = source.len();
        let leaf_node_count = item_count.div_ceil(B);

        let stem_node_count =
            Donnelly::<3, 64, 4, K>::get_stem_node_count_from_leaf_node_count(leaf_node_count);

        let max_stem_level: i32 = leaf_node_count.next_power_of_two().ilog2() as i32 - 1;

        // TODO: It would be nice to be able to determine the exact required length up-front.
        //  Instead, we just trim the stems afterwards by traversing right-child non-inf nodes
        //  till we hit max level to get the max used stem
        let stem_node_count = stem_node_count * Donnelly::<3, 64, 4, K>::stem_node_padding_factor();
        let stem_node_row_count = stem_node_count.div_ceil(B2);

        let mut stems = avec![[A::infinity(); B2]; stem_node_row_count];
        let mut leaf_points: [Vec<A>; K] = array_init(|_| Vec::with_capacity(item_count));
        let mut leaf_items: Vec<T> = Vec::with_capacity(item_count);
        let mut leaf_extents: Vec<(u32, u32)> = Vec::with_capacity(item_count.div_ceil(B));
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut sort_index = Vec::from_iter(0..item_count);

        if stem_node_count == 0 {
            // Write leaf and terminate recursion
            leaf_extents.push((0u32, sort_index.len() as u32));

            (0..sort_index.len()).for_each(|i| {
                (0..K).for_each(|dim| leaf_points[dim].push(source[sort_index[i]][dim]));
                leaf_items.push(sort_index[i].az::<T>())
            });
        } else {
            Self::populate_recursive(
                &mut stems,
                source,
                &mut sort_index,
                Donnelly::<3, 64, 4, K>::new(stems_ptr),
                max_stem_level,
                leaf_node_count * B,
                &mut leaf_points,
                &mut leaf_items,
                &mut leaf_extents,
            );

            // trim unneeded stems
            // TODO: eliminate the need for this
            // Donnelly::<3, 64, 4, K>::trim_unneeded_stems(&mut stems, max_stem_level as usize);
        }

        Self {
            stems,
            leaf_points,
            leaf_items,
            leaf_extents,
            max_stem_level,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_recursive(
        stems: &mut AVec<[A; B2], ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[[A; K]],
        sort_index: &mut [usize],
        mut stem_ordering: Donnelly<3, 64, 4, K>,
        max_stem_level: i32,
        capacity: usize,
        leaf_points: &mut [Vec<A>; K],
        leaf_items: &mut Vec<T>,
        leaf_extents: &mut Vec<(u32, u32)>,
    ) {
        let chunk_length = sort_index.len();
        let dim = stem_ordering.dim();

        if stem_ordering.level() > max_stem_level {
            // Write leaf and terminate recursion
            leaf_extents.push((
                leaf_items.len() as u32,
                (leaf_items.len() + chunk_length) as u32,
            ));

            (0..chunk_length).for_each(|i| {
                (0..K).for_each(|dim| leaf_points[dim].push(source[sort_index[i]][dim]));
                leaf_items.push(sort_index[i].az::<T>())
            });

            return;
        }

        let levels_below = max_stem_level - stem_ordering.level();
        let left_capacity = (2usize.pow(levels_below as u32) * B).min(capacity);
        let right_capacity = capacity.saturating_sub(left_capacity);

        let stem_index = stem_ordering.stem_idx();
        let mut pivot = Self::calc_pivot(chunk_length, stem_index, right_capacity);

        // only bother with this if we are putting at least one item in the right hand child
        if pivot < chunk_length {
            pivot = Self::update_pivot(source, sort_index, dim, pivot);

            // if we end up with a pivot of 0, something has gone wrong,
            // unless we only had a slice of len 1 anyway
            debug_assert!(pivot > 0 || chunk_length == 1);
            debug_assert!(
                stems[stem_index][0].is_infinite(), // TODO
                "Wrote to stem #{stem_index:?} for a second time",
            );

            // TODO
            stems[stem_index][0] = source[sort_index[pivot]][dim];
        }

        let right_stem_ordering = stem_ordering.branch();
        let (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);

        Self::populate_recursive(
            stems,
            source,
            lower_sort_index,
            stem_ordering,
            max_stem_level,
            left_capacity,
            leaf_points,
            leaf_items,
            leaf_extents,
        );

        Self::populate_recursive(
            stems,
            source,
            upper_sort_index,
            right_stem_ordering,
            max_stem_level,
            right_capacity,
            leaf_points,
            leaf_items,
            leaf_extents,
        );
    }

    #[cfg(not(feature = "unreliable_select_nth_unstable"))]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn update_pivot(
        source: &[[A; K]],
        sort_index: &mut [usize],
        dim: usize,
        mut pivot: usize,
    ) -> usize {
        // TODO: this block might be faster by using a quickselect with a fat partition?
        //       we could then run that quickselect and subtract (fat partition length - 1)
        //       from the pivot, avoiding the need for the while loop.

        // ensure the item whose index = pivot is in its correctly sorted position, and any
        // items that are equal to it are adjacent, according to our assumptions about the
        // behaviour of `select_nth_unstable_by` (See examples/check_select_nth_unstable.rs)
        sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));

        if pivot == 0 {
            return pivot;
        }

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 1 {
            pivot -= 1;
        }

        pivot
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use kiddo::immutable::float::kdtree::ImmutableKdBTree;
    /// use kiddo::Eytzinger;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdBTree<f64, u32, Eytzinger<3>, 3, 32> = ImmutableKdBTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn size(&self) -> usize {
        self.leaf_items.len()
    }

    /// Returns the theoretical max capacity of this tree
    #[inline]
    pub fn capacity(&self) -> usize {
        self.size()
    }

    fn calc_pivot(chunk_length: usize, _stem_index: usize, _right_capacity: usize) -> usize {
        chunk_length >> 1
    }

    /// Returns a LeafSlice for a given leaf index
    #[allow(unused)]
    #[inline]
    pub(crate) fn get_leaf_slice(&self, leaf_idx: usize) -> LeafSlice<'_, A, T, K> {
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

        // Artificially extend size to be at least chunk length for faster processing
        // TODO: why does this slow things down and break things?
        // let end = end.max(start + 32).min(self.leaf_items.len() as u32);

        LeafSlice::new(
            array_init::array_init(|i| &self.leaf_points[i][start as usize..end as usize]),
            &self.leaf_items[start as usize..end as usize],
        )
    }

    /// Returns the number of stem levels in the tree
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn max_stem_level(&self) -> i32 {
        self.max_stem_level
    }
}

#[cfg(test)]
mod tests {
    use crate::immutable::float::kdbtree::ImmutableKdBTree;
    use crate::SquaredEuclidean;
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};

    #[test]
    fn can_construct_an_empty_tree() {
        let tree = ImmutableKdBTree::<f64, u32, 3, 32, 8>::new_from_slice(&[]);
        let _result = tree.nearest_one::<SquaredEuclidean>(&[0.; 3]);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
        ];

        let _tree: ImmutableKdBTree<f32, usize, 2, 4, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);

        // assert_eq!(tree.leaf_extents[0].iter().count(), 3);
        // assert_eq!(tree.leaf_extents[1].iter().count(), 5);
        // assert_eq!(tree.leaf_extents[2].iter().count(), 4);
        // assert_eq!(tree.leaf_extents[3].iter().count(), 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_2() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];

        let _tree: ImmutableKdBTree<f32, usize, 2, 4, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_3() {
        use rand::seq::SliceRandom;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(23);

        let mut content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];
        content_to_add.shuffle(&mut rng);

        let _tree: ImmutableKdBTree<f32, usize, 2, 4, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);

        // assert_eq!(tree.leaves[0].size, 3);
        // assert_eq!(tree.leaves[1].size, 4);
        // assert_eq!(tree.leaves[2].size, 4);
        // assert_eq!(tree.leaves[3].size, 4);
        // assert_eq!(tree.leaves[4].size, 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_multiple_dupes() {
        use rand::seq::SliceRandom;

        for seed in 0..1_000 {
            //_000 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

            let mut content_to_add = vec![
                [4.0, 104.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [4.0, 104.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [17.0, 117.0],
                [18.0, 118.0],
            ];
            content_to_add.shuffle(&mut rng);

            let _tree: ImmutableKdBTree<f32, usize, 2, 8, 8> =
                ImmutableKdBTree::new_from_slice(&content_to_add);
        }
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_0() {
        let tree_size = 18;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);

        println!("tree: {tree:?}");
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_1() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_2() {
        let tree_size = 155;
        let seed = 480;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_3() {
        let tree_size = 26; // also 32
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_4() {
        let tree_size = 21;
        let seed = 131851;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_5() {
        let tree_size = 32;
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_6() {
        let tree_size = 56;
        let seed = 450533;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_7() {
        let tree_size = 18;
        let seed = 992063;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_8() {
        let tree_size = 19;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_9() {
        let tree_size = 20;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_10() {
        let tree_size = 36;
        let seed = 375096;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_11() {
        let tree_size = 10000;
        let seed = 257281;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_many_dupes() {
        let tree_size = 8;
        let seed = 0;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> =
            (0..tree_size).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut duped: Vec<[f32; 4]> = Vec::with_capacity(content_to_add.len() * 10);
        for item in content_to_add {
            for _ in 0..6 {
                duped.push(item);
            }
        }

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> = ImmutableKdBTree::new_from_slice(&duped);
    }

    #[test]
    fn can_construct_optimized_tree_medium_rand() {
        use itertools::Itertools;

        const TREE_SIZE: usize = 2usize.pow(19); // ~ 500k

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let num_uniq = content_to_add
            .iter()
            .flatten()
            .map(|&x| OrderedFloat(x))
            .unique()
            .count();

        println!("dupes: {:?}", TREE_SIZE * 4 - num_uniq);

        let _tree: ImmutableKdBTree<f32, usize, 4, 8, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let _tree: ImmutableKdBTree<f32, usize, 4, 32, 8> =
            ImmutableKdBTree::new_from_slice(&content_to_add);
    }
}
