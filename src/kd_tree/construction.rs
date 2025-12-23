use crate::kd_tree::{KdTree, StemLeafResolution};
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, Mutability, MutableLeafStrategy};
use crate::StemStrategy;
use aligned_vec::{avec, AVec, ConstAlign, CACHELINE_ALIGN};
use az::{Az, Cast};
use nonmax::NonMaxUsize;
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: MutableLeafStrategy<A, T, SS, K, B>,
{
    /// Adds a point and associated item to the tree.
    ///
    /// If the target leaf is full, it will be split before insertion.
    pub fn add(&mut self, point: &[A; K], item: T) {
        // Find the target leaf
        let (stem_strat, parent_stem_idx, is_right_child) = self.find_leaf_with_context(point);
        let leaf_idx = stem_strat.leaf_idx();

        if !self.leaves.is_leaf_full(leaf_idx) {
            self.leaves.add_to_leaf(leaf_idx, point, item);
            self.size += 1;
            return;
        }

        // Leaf is full, need to split
        let is_first_split = self.leaves.leaf_count() == 1;
        let (pivot_val, split_dim, new_leaf_idx) =
            self.split_leaf(stem_strat, parent_stem_idx, is_right_child, is_first_split);

        // determine which leaf we belong in after the split
        let leaf_idx = if point[split_dim] >= pivot_val {
            new_leaf_idx
        } else {
            leaf_idx
        };

        self.leaves.add_to_leaf(leaf_idx, point, item);
        self.size += 1;
    }

    /// Find the leaf for a query point, along with context needed for splitting.
    /// Returns: (stem_strategy, parent_stem_idx, is_right_child)
    fn find_leaf_with_context(&self, query: &[A; K]) -> (SS, Option<NonMaxUsize>, bool) {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);
        let mut parent_stem_idx: Option<NonMaxUsize> = None;
        let mut is_right_child = false;

        while stem_strat.level() <= self.max_stem_level {
            let stem_idx = stem_strat.stem_idx();

            // Check if this stem points directly to a leaf (only for Mapped)
            if let Some(_leaf_idx) = self.resolve_terminal_stem(stem_idx) {
                return (stem_strat, parent_stem_idx, is_right_child);
            }

            parent_stem_idx = Some(NonMaxUsize::new(stem_idx).unwrap());
            let pivot = unsafe { self.stems.get_unchecked(stem_idx) };
            is_right_child = unsafe { *query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        (stem_strat, parent_stem_idx, is_right_child)
    }

    /// Split a full leaf, moving some of the points in the existing leaf to a new one.
    /// Updates the stem tree to contain the new pivot value, pointing to the existing and
    /// split-off leaf.
    ///
    /// Returns the dimension along which the split occurred and the value of the pivot, as well
    /// as the new leaf index.
    fn split_leaf(
        &mut self,
        stem_strategy: SS,
        parent_stem_idx: Option<NonMaxUsize>,
        is_right_child: bool,
        is_first_split: bool,
    ) -> (A, usize, usize) {
        let old_leaf_idx = stem_strategy.leaf_idx();
        let split_dim = stem_strategy.dim();

        // Split the leaf
        let (pivot_val, new_leaf_idx) = self.leaves.split_leaf(old_leaf_idx, split_dim);

        // Get the child indices where the two leaves will be pointed to
        let (left_child_idx, right_child_idx) = stem_strategy.child_indices();

        let stem_idx = if is_first_split {
            // First split: update the root stem from max_value to the actual pivot
            // Use parent_stem_idx if available, otherwise get the root index

            // Overwrite the initial dummy pivot value with the actual pivot
            // (We use a dummy pivot value like this so that the branching logic needed to
            // deal with an empty stem tree is only needed on a split rather than on a query)
            SS::new(NonNull::dangling()).stem_idx()
        } else {
            stem_strategy.stem_idx()
        };

        self.stems[stem_idx] = pivot_val;

        // Update the leaf_idx_map to point children to the two leaves
        if let StemLeafResolution::Mapped { leaf_idx_map, .. } = &mut self.stem_leaf_resolution {
            // Ensure the map is large enough
            if leaf_idx_map.len() < right_child_idx + 1 {
                leaf_idx_map.resize(right_child_idx + 1, None);
            }

            // Clear the root's mapping (it's now an interior node, not a leaf)
            leaf_idx_map[stem_idx] = None;

            // Map left child to old leaf, right child to new leaf
            leaf_idx_map[left_child_idx] = NonMaxUsize::new(old_leaf_idx);
            leaf_idx_map[right_child_idx] = NonMaxUsize::new(new_leaf_idx);
        }

        // Increment max_stem_level since we now have children
        self.max_stem_level += 1;

        (pivot_val, split_dim, new_leaf_idx)
    }

    /// Transition from Pristine to Mapped state on first split
    #[allow(unused)]
    fn taint_if_pristine(
        &mut self,
        new_stem_idx: usize,
        _left_leaf_idx: usize,
        _right_leaf_idx: usize,
        parent_stem_idx: Option<usize>,
    ) {
        match &self.stem_leaf_resolution {
            StemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } => {
                // Transition to Mapped
                let min_stem_leaf_idx = 1 << *stems_depth;
                let mut leaf_idx_map = vec![None; self.stems.len()];

                // Map all existing leaves using arithmetic
                for i in 0..*leaf_count {
                    let stem_idx = min_stem_leaf_idx + i;
                    if stem_idx < leaf_idx_map.len() {
                        leaf_idx_map[stem_idx - min_stem_leaf_idx] = NonMaxUsize::new(i);
                    }
                }

                // Update mapping for the new stem and leaves
                if let Some(parent_idx) = parent_stem_idx {
                    // Clear parent's mapping (it now has children)
                    if parent_idx >= min_stem_leaf_idx {
                        leaf_idx_map[parent_idx - min_stem_leaf_idx] = None;
                    }
                }

                // New stem points to the two leaves
                if new_stem_idx >= min_stem_leaf_idx {
                    let idx = new_stem_idx - min_stem_leaf_idx;
                    if idx >= leaf_idx_map.len() {
                        leaf_idx_map.resize(idx + 1, None);
                    }
                }

                self.stem_leaf_resolution = StemLeafResolution::Mapped {
                    min_stem_leaf_idx,
                    leaf_idx_map,
                };
            }
            StemLeafResolution::Mapped { .. } => {
                // Already mapped, just update the mapping
                // TODO: implement mapping updates
            }
            _ => {
                // Arithmetic/Immutable - should not be calling this
                panic!("Cannot split leaves in immutable tree");
            }
        }
    }

    /// Removes a point and associated item from the tree.
    ///
    /// Note: This does not rebalance the tree.
    pub fn remove(&mut self, point: &[A; K], item: T) {
        let leaf_idx = self.get_leaf_idx(point);

        self.leaves.remove_from_leaf(leaf_idx, point, item);

        // TODO: attempt to prune leaf if now empty
    }
}

// Shared construction implementation (works for both Immutable and Mutable)
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
        Self::new_from_slice_with(source, |leaf_items: &mut Vec<T>, src_idx: usize| {
            leaf_items.push(src_idx.az::<T>());
        })
    }

    /// Inner constructor shared by all variants. The `push_item` callback
    /// is invoked wherever we would normally push an item for a source index.
    ///
    /// This has *no* `usize: Cast<T>` bound; callers can decide what to do
    /// with the index (e.g. cast it, ignore it, or something else).
    fn new_from_slice_with<F>(source: &[[A; K]], mut push_item: F) -> Self
    where
        F: FnMut(&mut Vec<T>, usize),
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
                push_item(&mut leaf_items, src_idx);
            }

            let leaf_points_refs: [&[A]; K] =
                array_init::array_init(|dim| leaf_points[dim].as_slice());

            leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());
        } else {
            Self::populate_recursive_with(
                &mut stems,
                source,
                &mut sort_index,
                SS::new(stems_ptr),
                max_stem_level,
                leaf_node_count * B,
                &mut leaves,
                &mut push_item,
            );

            // TODO: eliminate the need for this
            SS::trim_unneeded_stems(&mut stems, max_stem_level as usize);
        }

        // Initialize stem-to-leaf resolution strategy based on leaf mutability
        let stem_leaf_resolution =
            LS::Mutability::initial_stem_leaf_resolution(max_stem_level as usize, leaf_node_count);

        Self {
            stems,
            leaves,
            stem_leaf_resolution,
            size: item_count,
            max_stem_level,
            _phantom: Default::default(),
        }
    }
}

impl<A, SS, LS, const K: usize, const B: usize> KdTree<A, (), SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    SS: StemStrategy,
    LS: LeafStrategy<A, (), SS, K, B>,
{
    /// Creates a `KdTree` with no stored item values (`T = ()`).
    ///
    /// Leaf item slices will have the correct length but contain only `()`.
    /// LLVM can generally optimize the `Vec<()>` storage away.
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn new_from_slice_no_items(source: &[[A; K]]) -> Self {
        Self::new_from_slice_with(source, |leaf_items: &mut Vec<()>, _src_idx: usize| {
            leaf_items.push(());
        })
    }
}

// Shared utility methods for construction (available to both Immutable and Mutable)
impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    /// Shared recursive tree construction helper
    #[allow(clippy::too_many_arguments)]
    fn populate_recursive_with<F>(
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[[A; K]],
        sort_index: &mut [usize],
        mut stem_ordering: SS,
        max_stem_level: i32,
        capacity: usize,
        leaves: &mut LS,
        push_item: &mut F,
    ) where
        F: FnMut(&mut Vec<T>, usize),
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
                for d in 0..K {
                    leaf_points[d].push(source[src_idx][d]);
                }
                // delegate item handling
                push_item(&mut leaf_items, src_idx);
            }

            // Convert [Vec<A>; K] -> [&[A]; K]
            let leaf_points_refs: [&[A]; K] = array_init::array_init(|d| leaf_points[d].as_slice());

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

        Self::populate_recursive_with(
            stems,
            source,
            lower_sort_index,
            stem_ordering,
            max_stem_level,
            left_capacity,
            leaves,
            push_item,
        );

        Self::populate_recursive_with(
            stems,
            source,
            upper_sort_index,
            right_stem_ordering,
            max_stem_level,
            right_capacity,
            leaves,
            push_item,
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
