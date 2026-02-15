use crate::kd_tree::{KdTree, StemLeafResolution};
use crate::traits_unified_2::{
    AxisUnified, Basics, BucketLimitType, LeafStrategy, Mutability, MutableLeafStrategy,
};
use crate::StemStrategy;
use aligned_vec::{avec, AVec, ConstAlign, CACHELINE_ALIGN};
use az::{Az, Cast};
use nonmax::NonMaxUsize;
use std::fmt::Display;
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq + Display,
    SS: StemStrategy,
    LS: MutableLeafStrategy<A, T, SS, K, B>,
{
    /// Adds a point and associated item to the tree.
    ///
    /// If the target leaf is full, it will be split before insertion.
    pub fn add(&mut self, point: &[A; K], item: T) {
        // Find the target leaf
        let (stem_strat, parent_stem_idx, is_right_child) = self.find_leaf_with_context(point);
        let leaf_idx = match &self.stem_leaf_resolution {
            StemLeafResolution::Mapped { leaf_idx_map, .. } => {
                leaf_idx_map[stem_strat.stem_idx()].unwrap().get()
            }
            _ => stem_strat.leaf_idx(),
        };

        if !self.leaves.is_leaf_full(leaf_idx) {
            self.leaves.add_to_leaf(leaf_idx, point, item);
            self.size += 1;
            return;
        }

        // println!("Leaf {leaf_idx} is full, splitting. {self}");

        // Leaf is full, need to split
        let (pivot_val, split_dim, new_leaf_idx) =
            self.split_leaf(leaf_idx, stem_strat, parent_stem_idx, is_right_child);

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

    /// Split a full leaf, moving some points in the existing leaf to a new one.
    /// Updates the stem tree to contain the new pivot value, pointing to the existing and
    /// split-off leaf.
    ///
    /// Returns the dimension along which the split occurred and the value of the pivot, as well
    /// as the new leaf index.
    fn split_leaf(
        &mut self,
        leaf_idx: usize,
        stem_strategy: SS,
        _parent_stem_idx: Option<NonMaxUsize>,
        _is_right_child: bool,
    ) -> (A, usize, usize) {
        let old_leaf_idx = leaf_idx; // stem_strategy.leaf_idx();
        let split_dim = stem_strategy.dim();

        // Split the leaf
        let (pivot_val, new_leaf_idx) = self.leaves.split_leaf(old_leaf_idx, split_dim);

        // Get the indices of the children of the stem at which the split occurs
        let (left_child_idx, right_child_idx) = stem_strategy.child_indices();
        let stem_idx = stem_strategy.stem_idx();

        // Ensure the stem array is large enough
        if self.stems.len() < stem_idx + 1 {
            self.stems.resize(stem_idx + 1, A::max_value());
        }

        self.stems[stem_idx] = pivot_val;

        // Update the leaf_idx_map to point children to the two leaves
        if let StemLeafResolution::Mapped { leaf_idx_map, .. } = &mut self.stem_leaf_resolution {
            // Ensure the map is large enough
            if leaf_idx_map.len() < right_child_idx + 1 {
                leaf_idx_map.resize(right_child_idx + 1, None);
            }

            // Map left child to old leaf
            leaf_idx_map[left_child_idx] = leaf_idx_map[stem_idx];
            // Clear the root's mapping (it's now an interior node, not a leaf)
            leaf_idx_map[stem_idx] = None;
            // Map right child to new leaf
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
    /// use kiddo::kd_tree::KdTree;
    /// use kiddo::kd_tree::leaf_strategies::FlatVec;
    /// use kiddo::Eytzinger;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: KdTree<f64, u32, Eytzinger<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
    ///     KdTree::new_from_slice(&points);
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

        if leaf_node_count < 2 {
            return Self::new_from_slice_no_stems_with(source, push_item);
        }

        let mut stems_depth: usize = leaf_node_count.next_power_of_two().ilog2() as usize;

        // Pad stem tree height to the next block boundary for block-based strategies
        let padding_level_count = if !stems_depth.is_multiple_of(SS::block_size()) {
            let padding_level_count = SS::block_size() - (stems_depth % SS::block_size());
            stems_depth += padding_level_count;
            padding_level_count
        } else {
            0
        };

        // Padding levels will be placed at the root of the tree. Pre-traverse any padding levels
        // so that stem_strat is set to the location where the true root will be
        let mut stem_strat = SS::new_no_ptr();
        for _ in 0..padding_level_count {
            stem_strat.traverse(false);
        }
        let root_stem_strat = stem_strat.clone();

        // Traverse to the right-most leaf to determine the max used stem index
        let rightmost_leaf_idx = leaf_node_count - 1;
        for bit_idx in (1..stems_depth).rev() {
            let is_right = rightmost_leaf_idx & (1 << bit_idx) != 0;
            stem_strat.traverse(is_right);
        }
        let stem_node_count = stem_strat.stem_idx() + 1;

        // rounded up to the nearest multiple of 8 if not a multiple of 8 already
        let stem_node_count_padded = stem_node_count.div_ceil(8) * 8;
        let mut stems = avec![A::max_value(); stem_node_count_padded];

        let mut leaves = LS::new_with_capacity(item_count);
        let mut terminal_stem_indices = Vec::with_capacity(leaf_node_count);
        let mut actual_max_stem_level: i32 = -1;
        let mut sort_index = Vec::from_iter(0..item_count);

        Self::populate_recursive_with(
            &mut stems,
            source,
            &mut sort_index,
            root_stem_strat,
            stems_depth as i32 - 1,
            leaf_node_count * B,
            &mut leaves,
            &mut terminal_stem_indices,
            &mut actual_max_stem_level,
            &mut push_item,
        );

        let initial_max_stem_level = stems_depth as i32 - 1;
        let stem_leaf_resolution =
            if LS::Mutability::is_mutable() || actual_max_stem_level > initial_max_stem_level {
                Self::mapped_stem_leaf_resolution_from_terminals(&terminal_stem_indices)
            } else {
                LS::Mutability::initial_stem_leaf_resolution::<SS>(stems_depth, leaf_node_count)
            };

        // println!("Stems: {:?}", &stems);
        Self {
            stems,
            leaves,
            stem_leaf_resolution,
            size: item_count,
            max_stem_level: actual_max_stem_level,
            _phantom: Default::default(),
        }
    }

    fn new_from_slice_no_stems_with<F>(source: &[[A; K]], mut push_item: F) -> Self
    where
        F: FnMut(&mut Vec<T>, usize),
    {
        let item_count = source.len();

        if item_count == 0 {
            return Self::default();
        }

        let mut leaf_points: [Vec<A>; K] =
            array_init::array_init(|_| Vec::with_capacity(item_count));
        let mut leaf_items: Vec<T> = Vec::with_capacity(item_count);

        for idx in 0..item_count {
            for dim in 0..K {
                leaf_points[dim].push(source[idx][dim]);
            }
            push_item(&mut leaf_items, idx);
        }

        let leaf_points_refs: [&[A]; K] = array_init::array_init(|dim| leaf_points[dim].as_slice());

        let mut leaves = LS::new_with_capacity(item_count);
        leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());

        let stem_leaf_resolution =
            LS::Mutability::initial_stem_leaf_resolution::<SS>(0, leaves.leaf_count());

        Self {
            stems: avec![A::max_value(); 0],
            leaves,
            stem_leaf_resolution,
            size: item_count,
            max_stem_level: -1,
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
    fn at_leaf_level(
        current_stem_level: i32,
        max_stem_level: i32,
        bucket_limit_type: BucketLimitType,
        chunk_length: usize,
        capacity: usize,
    ) -> bool {
        match bucket_limit_type {
            // TODO: check that this is the behaviour that we want
            BucketLimitType::Soft => current_stem_level > max_stem_level || capacity <= B,
            BucketLimitType::Hard => chunk_length <= B,
        }
    }

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
        terminal_stem_indices: &mut Vec<usize>,
        actual_max_stem_level: &mut i32,
        push_item: &mut F,
    ) where
        F: FnMut(&mut Vec<T>, usize),
    {
        let chunk_length = sort_index.len();
        let dim = stem_ordering.construction_dim();

        debug_assert!(
            chunk_length > 0,
            "recursed an empty chunk (stem_idx={}, level={}, chunk_length={}, capacity={})",
            stem_ordering.stem_idx(),
            stem_ordering.level(),
            chunk_length,
            capacity,
        );

        if Self::at_leaf_level(
            stem_ordering.level(),
            max_stem_level,
            LS::BUCKET_LIMIT_TYPE,
            chunk_length,
            capacity,
        ) {
            // Write leaf and terminate recursion
            let leaf_len = sort_index.len();

            debug_assert!(
                leaf_len > 0,
                "attempted to construct an empty leaf (stem_idx={}, level={}, capacity={})",
                stem_ordering.stem_idx(),
                stem_ordering.level(),
                capacity
            );
            let mut leaf_points: [Vec<A>; K] =
                array_init::array_init(|_| Vec::with_capacity(leaf_len));
            let mut leaf_items: Vec<T> = Vec::with_capacity(leaf_len);

            for &src_idx in sort_index.iter() {
                for d in 0..K {
                    leaf_points[d].push(source[src_idx][d]);
                }
                push_item(&mut leaf_items, src_idx);
            }

            // Convert [Vec<A>; K] -> [&[A]; K]
            let leaf_points_refs: [&[A]; K] = array_init::array_init(|d| leaf_points[d].as_slice());

            leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());
            terminal_stem_indices.push(stem_ordering.stem_idx());
            return;
        }

        let levels_below = max_stem_level - stem_ordering.level();
        let clamped_levels_below = levels_below.max(0) as u32;
        let left_capacity = (2usize.pow(clamped_levels_below) * B).min(capacity);
        let right_capacity = capacity.saturating_sub(left_capacity);

        debug_assert!(
            left_capacity > 0,
            "left_capacity is zero - should never happen (stem_idx={}, level={}, chunk_length={}, capacity={}, right_capacity={})",
            stem_ordering.stem_idx(),
            stem_ordering.level(),
            chunk_length,
            capacity,
            right_capacity
        );

        let stem_index = stem_ordering.stem_idx();
        *actual_max_stem_level = (*actual_max_stem_level).max(stem_ordering.level());

        if stem_index >= stems.len() {
            tracing::warn!(
                %stem_index,
                existing_stem_vec_len = %stems.len(),
                "encountered a stem index beyond the end of the stem vec. Growing the vec to fit"
            );

            stems.resize(stem_index + 1, A::max_value());
        }

        let mut pivot = Self::calc_pivot(
            chunk_length,
            stem_index,
            right_capacity,
            LS::BUCKET_LIMIT_TYPE,
        );

        debug_assert!(
            pivot > 0,
            "construction produced initial pivot=0 (empty-left split candidate): \
            stem_index = {}, level={}, chunk_length = {}, capacity = {}, \
            left_capacity = {}, right_capacity = {}, dim = {}",
            stem_index,
            stem_ordering.level(),
            chunk_length,
            capacity,
            left_capacity,
            right_capacity,
            dim,
        );

        // only bother with this logic if we are putting at least one item in the right-hand child
        if pivot < chunk_length {
            pivot = Self::update_pivot(source, sort_index, dim, pivot);

            debug_assert!(
                pivot > 0,
                "construction produced updated pivot=0 (empty-left split candidate): \
                stem_index = {}, level={}, chunk_length = {}, capacity = {}, \
                left_capacity = {}, right_capacity = {}, dim = {}",
                stem_index,
                stem_ordering.level(),
                chunk_length,
                capacity,
                left_capacity,
                right_capacity,
                dim,
            );

            // if we end up with a pivot of 0, something has gone wrong,
            // unless we only had a slice of len 1 anyway
            // debug_assert!(
            //     pivot > 0 || chunk_length == 1,
            // );

            // if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Hard {
            //     debug_assert!(
            //         right_capacity >= chunk_length.saturating_sub(pivot),
            //         "right_capacity ({right_capacity}) should be greater than chunk_length - pivot ({chunk_length} - {pivot})"
            //     );
            // }

            if pivot < chunk_length {
                debug_assert!(
                    A::Coord::is_max_value(stems[stem_index]),
                    "Wrote to stem #{stem_index:?} for a second time",
                );

                stems[stem_index] = source[sort_index[pivot]][dim];
            }
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
            terminal_stem_indices,
            actual_max_stem_level,
            push_item,
        );

        if !upper_sort_index.is_empty() {
            Self::populate_recursive_with(
                stems,
                source,
                upper_sort_index,
                right_stem_ordering,
                max_stem_level,
                right_capacity,
                leaves,
                terminal_stem_indices,
                actual_max_stem_level,
                push_item,
            );
        }
    }

    // TODO: remove this entirely in favor of just taking ownership of terminal_stem_indices
    //       once confident that the debug_asserts never fire
    fn mapped_stem_leaf_resolution_from_terminals(
        terminal_stem_indices: &[usize],
    ) -> StemLeafResolution {
        if terminal_stem_indices.is_empty() {
            return StemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map: Vec::new(),
            };
        }

        // TODO: this should not be needed. Just use terminal_stem_indices.len()
        let max_terminal_stem_idx = terminal_stem_indices.iter().copied().max().unwrap_or(0);
        // debug_assert!(
        //     max_terminal_stem_idx == terminal_stem_indices.len() - 1,
        //     "Leaf array should be contiguous. Construction invariant failed"
        // );
        if max_terminal_stem_idx > terminal_stem_indices.len() - 1 {
            tracing::warn!("Leaf array should be contiguous. Construction invariant failed");
        };
        let mut leaf_idx_map: Vec<Option<NonMaxUsize>> = vec![None; max_terminal_stem_idx + 1];

        for (leaf_idx, &terminal_stem_idx) in terminal_stem_indices.iter().enumerate() {
            debug_assert!(
                leaf_idx_map[terminal_stem_idx].is_none(),
                "Duplicate terminal stem index in mapped leaf_idx_map construction: stem_idx={} existing_leaf_idx={} new_leaf_idx={}",
                terminal_stem_idx,
                leaf_idx_map[terminal_stem_idx].unwrap(),
                leaf_idx
            );

            leaf_idx_map[terminal_stem_idx] = NonMaxUsize::new(leaf_idx);
        }

        StemLeafResolution::Mapped {
            min_stem_leaf_idx: 0,
            leaf_idx_map,
        }
    }

    fn calc_pivot(
        chunk_length: usize,
        _stem_index: usize,
        right_capacity: usize,
        bucket_limit_type: BucketLimitType,
    ) -> usize {
        let mut result = chunk_length
            .saturating_sub(right_capacity)
            .next_multiple_of(B)
            .min(chunk_length);

        // Treat this as the ideal split target: put at least B items on the left
        // whenever the chunk can support it.
        if chunk_length > 0 {
            let min_ideal_left = B.min(chunk_length);
            if result < min_ideal_left {
                result = min_ideal_left;
            }
        }

        // debug_assert!(
        //     result > 0 || chunk_length >= right_capacity,
        //     "Unexpectedly generated an initial pivot to split a slice at position 0 during construction (chunk length: {chunk_length}, right_capacity: {right_capacity})"
        // );

        let result = if bucket_limit_type == BucketLimitType::Hard
            && result >= chunk_length
            && result > B
        {
            let adjusted_result = chunk_length
                .saturating_sub(right_capacity.max(B))
                .next_multiple_of(B)
                .min(chunk_length);

            tracing::debug!(
                orig_pivot = %result,
                adjusted_pivot = %adjusted_result,
                %chunk_length,
                %right_capacity,
                "initial pivot calc would result in infinite recursion due to everything going in left but left being > B. Splitting finer"
            );

            adjusted_result
        } else {
            result
        };

        debug_assert!(
            result < chunk_length + 1,
            "Unexpectedly generated an initial pivot to split a slice at or beyond its end during construction (chunk length: {chunk_length}, right_capacity: {right_capacity})"
        );

        result
    }

    // #[cfg(not(feature = "unreliable_select_nth_unstable"))]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn update_pivot(
        source: &[[A; K]],
        sort_index: &mut [usize],
        dim: usize,
        init_pivot: usize,
    ) -> usize {
        // TODO: this block might be faster by using a quickselect with a fat partition?
        //       we could then run that quickselect and subtract (fat partition length - 1)
        //       from the pivot, avoiding the need for the while loop.

        let mut pivot = init_pivot;

        // ensure the item whose index = pivot is in its correctly sorted position, and any
        // items that are equal to it are adjacent, according to our assumptions about the
        // behaviour of `select_nth_unstable_by` (See examples/check_select_nth_unstable.rs)
        sort_index
            .select_nth_unstable_by(pivot, |&ia, &ib| A::cmp(source[ia][dim], source[ib][dim]));

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while pivot > 0 && source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] {
            pivot -= 1;
        }

        // if we nudged it all the way to the left, reset and try nudging it rightwards from the
        // initial pivot point instead. This requires that the entire slice is sorted, rather than
        // just the left-hand side
        if pivot == 0 {
            pivot = init_pivot;

            sort_index.sort_unstable_by(|&ia, &ib| A::cmp(source[ia][dim], source[ib][dim]));

            while pivot < sort_index.len()
                && source[sort_index[pivot]][dim] == source[sort_index[pivot + 1]][dim]
            {
                pivot += 1;
            }

            if pivot == sort_index.len() {
                // if we end up here with pivot == sort_index.len(), then the source slice is unsplittable
                // in this dimension due to all entries having the same value on the given dimension
                tracing::debug!(
                    slice_len = %sort_index.len(),
                    %dim,
                    "Slice unsplittable along dimension"
                );

                if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Hard {
                    panic!(
                        "Slice unsplittable along dimension. More points with the same value along \
                        splitting dimension than will fit in a bucket"
                    );
                }
            } else {
                // Avoid empty-left splits: place the boundary after the run.
                pivot += 1;
                tracing::trace!(
                    slice_len = %sort_index.len(),
                    %dim,
                    %init_pivot,
                    shift = %(pivot - init_pivot),
                    %pivot,
                    "pivot shifted right"
                );
            }
        } else if pivot != init_pivot {
            tracing::trace!(
                slice_len = %sort_index.len(),
                %dim,
                %init_pivot,
                shift = %(init_pivot - pivot),
                %pivot,
                "pivot shifted left"
            );
        }

        pivot
    }
}
