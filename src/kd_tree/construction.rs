use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, MutableLeafStrategy};
use crate::StemStrategy;
use aligned_vec::{avec, AVec, ConstAlign, CACHELINE_ALIGN};
use az::{Az, Cast};
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    SS: StemStrategy,
    LS: MutableLeafStrategy<A, T, SS, K, B>,
{
    pub fn add(&mut self, point: &[A; K], item: T) {
        // get matching leaf idx by traversal

        // is leaf full?
        // * perform split, getting a new stem_idx
        // * traverse from that new stem_idx to update leaf_idx to new match

        // Get leaf. Insert point & item.
        // Update leaf and tree size

        let leaf_idx = self.get_leaf_idx(point);

        if !self.leaves.is_leaf_full(leaf_idx) {
            self.leaves.add_to_leaf(leaf_idx, point, item);
            self.size += 1;
            return;
        }

        self.leaves.split_leaf(leaf_idx);

        // TODO: more efficient to navigate from leaf_idx rather than root
        let leaf_idx = self.get_leaf_idx(point);

        self.leaves.add_to_leaf(leaf_idx, point, item);
        self.size += 1;
    }

    pub fn remove(&mut self, point: &[A; K], item: T) {
        let leaf_idx = self.get_leaf_idx(point);
        self.leaves.remove_from_leaf(leaf_idx, point, item);
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    /// Creates a `KdTree`, balanced and optimised, populated
    /// with items from `source`.
    ///
    /// `KdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::kdtree::KdTree;
    /// use kiddo::Eytzinger;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: KdTree<f64, u32, Eytzinger<3>, 3, 32> = KdTree::new_from_slice(&points);
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
        let stem_node_count = SS::get_stem_node_count_from_leaf_node_count(leaf_node_count);
        let max_stem_level: i32 = leaf_node_count.next_power_of_two().ilog2() as i32 - 1;

        // TODO: It would be nice to be able to determine the exact required length up-front.
        //  Instead, we just trim the stems afterwards by traversing right-child non-inf nodes
        //  till we hit max level to get the max used stem
        let stem_node_count = stem_node_count * SS::stem_node_padding_factor();

        let mut stems = avec![A::max_value(); stem_node_count];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut leaves = LS::new_with_capacity(item_count);
        let mut sort_index = Vec::from_iter(0..item_count);

        if stem_node_count == 0 {
            // Special case: no stems needed, so we can just write the leaf directly.
            let leaf_len = sort_index.len();
            let mut leaf_points: [Vec<A>; K] =
                array_init::array_init(|_| Vec::with_capacity(leaf_len));
            let mut leaf_items: Vec<T> = Vec::with_capacity(leaf_len);

            for &src_idx in sort_index.iter() {
                for dim in 0..K {
                    leaf_points[dim].push(source[src_idx][dim]);
                }
                leaf_items.push(src_idx.az::<T>());
            }

            // Convert to the shape expected by LeafStrategy::append_leaf: &[&[A]; K]
            let leaf_points_refs: [&[A]; K] =
                array_init::array_init(|dim| leaf_points[dim].as_slice());

            leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());
        } else {
            Self::populate_recursive(
                &mut stems,
                source,
                &mut sort_index,
                SS::new(stems_ptr),
                max_stem_level,
                leaf_node_count * B,
                &mut leaves,
            );

            // TODO: eliminate the need for this
            SS::trim_unneeded_stems(&mut stems, max_stem_level as usize);
        }

        Self {
            stems,
            leaves,
            size: item_count,
            max_stem_level,
            _phantom: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_recursive(
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[[A; K]],
        sort_index: &mut [usize],
        mut stem_ordering: SS,
        max_stem_level: i32,
        capacity: usize,
        leaves: &mut LS,
    ) where
        usize: Cast<T>,
    {
        let chunk_length = sort_index.len();
        let dim = stem_ordering.dim();

        if stem_ordering.level() > max_stem_level {
            // Write leaf and terminate recursion
            let leaf_len = sort_index.len();
            let mut leaf_points: [Vec<A>; K] =
                array_init::array_init(|_| Vec::with_capacity(leaf_len));
            let mut leaf_items: Vec<T> = Vec::with_capacity(leaf_len);

            // Gather from `source` via `sort_index`
            for &src_idx in sort_index.iter() {
                // points
                for dim in 0..K {
                    leaf_points[dim].push(source[src_idx][dim]);
                }
                // item index (or whatever mapping you want)
                leaf_items.push(src_idx.az::<T>());
            }

            // Convert [Vec<A>; K] -> [&[A]; K]
            let leaf_points_refs: [&[A]; K] =
                array_init::array_init(|dim| leaf_points[dim].as_slice());

            leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());
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
                A::Coord::is_max_value(stems[stem_index]),
                "Wrote to stem #{stem_index:?} for a second time",
            );

            stems[stem_index] = source[sort_index[pivot]][dim];
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
            leaves,
        );

        Self::populate_recursive(
            stems,
            source,
            upper_sort_index,
            right_stem_ordering,
            max_stem_level,
            right_capacity,
            leaves,
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
        sort_index
            .select_nth_unstable_by(pivot, |&ia, &ib| A::cmp(source[ia][dim], source[ib][dim]));

        if pivot == 0 {
            return pivot;
        }

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 1 {
            pivot -= 1;
        }

        pivot
    }

    fn calc_pivot(chunk_length: usize, _stem_index: usize, _right_capacity: usize) -> usize {
        chunk_length >> 1
    }
}
