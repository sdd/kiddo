use crate::kd_tree::leaf_view::LeafArena;
use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{
    AxisUnified, Basics, BucketLimitType, ConstructibleLeafStrategy, Immutable, LeafProjection,
    LeafStrategy,
};
use crate::StemStrategy;
use aligned_vec::{AVec, CACHELINE_ALIGN};

/// Immutable leaf storage using chunk-tiled arenas encoded into a single byte buffer.
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
pub struct VecOfArenas<A, T, const K: usize, const B: usize> {
    leaf_extents: Vec<(usize, usize)>,
    #[cfg_attr(
        feature = "rkyv_08",
        rkyv(with = crate::rkyv_08_impl::AsAlignedCachelineABox)
    )]
    leaf_bytes: AVec<u8>,
    size: usize,
    _phantom: std::marker::PhantomData<(A, T)>,
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize> ArchivedVecOfArenas<A, T, K, B>
where
    A: rkyv_08::Archive + Copy,
    T: rkyv_08::Archive + Copy,
{
    /// Returns the number of items in a leaf.
    #[inline]
    pub fn leaf_len(&self, leaf_idx: usize) -> usize {
        self.leaf_extents[leaf_idx].1.to_native() as usize
    }

    /// Returns an arena-backed view over a leaf.
    #[inline]
    pub fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, A, T, K> {
        let extent = &self.leaf_extents[leaf_idx];
        let offset = extent.0.to_native() as usize;
        let len = extent.1.to_native() as usize;
        let bytes = self.leaf_bytes.get().as_slice();
        LeafArena::new(unsafe { bytes.as_ptr().add(offset) }, len)
    }
}

#[cfg(feature = "rkyv_08")]
impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for ArchivedVecOfArenas<AX, T, K, B>
where
    AX: rkyv_08::Archive + AxisUnified<Coord = AX>,
    T: rkyv_08::Archive + Basics,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafArena;

    fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        ArchivedVecOfArenas::leaf_len(self, leaf_idx)
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!("VecOfArenas currently exposes only arena-backed hot paths")
    }

    fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, AX, T, K> {
        ArchivedVecOfArenas::leaf_arena(self, leaf_idx)
    }
}

#[cfg(test)]
#[inline(always)]
fn extend_bytes_from_slice<U: Copy>(dst: &mut Vec<u8>, src: &[U]) {
    let byte_len = std::mem::size_of_val(src);
    let start = dst.len();
    dst.reserve(byte_len);

    unsafe {
        dst.set_len(start + byte_len);
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr().add(start),
            byte_len,
        );
    }
}

#[inline(always)]
fn extend_avec_bytes_from_slice<U: Copy>(dst: &mut AVec<u8>, src: &[U]) {
    let byte_len = std::mem::size_of_val(src);
    let start = dst.len();
    dst.reserve(byte_len);

    unsafe {
        dst.set_len(start + byte_len);
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr().add(start),
            byte_len,
        );
    }
}

#[inline(always)]
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for VecOfArenas<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafArena;

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        debug_assert!(leaf_idx < self.leaf_extents.len());
        unsafe { self.leaf_extents.get_unchecked(leaf_idx).1 }
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!("VecOfArenas currently exposes only arena-backed hot paths")
    }

    fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, AX, T, K> {
        debug_assert!(leaf_idx < self.leaf_extents.len());
        let (offset, len) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        #[cfg(debug_assertions)]
        {
            let byte_len = LeafArena::<AX, T, K>::encoded_len_bytes(len);
            debug_assert!(offset + byte_len <= self.leaf_bytes.len());
        }

        LeafArena::new(unsafe { self.leaf_bytes.as_ptr().add(offset) }, len)
    }

    #[inline]
    fn maybe_enable_huge_pages(&self) {
        crate::huge_pages::maybe_collapse_slice_huge_pages(
            self.leaf_extents.as_ptr(),
            self.leaf_extents.len(),
        );
        crate::huge_pages::maybe_collapse_slice_huge_pages(
            self.leaf_bytes.as_ptr(),
            self.leaf_bytes.len(),
        );
    }
}

impl<AX, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<AX, T, SS, K, B>
    for VecOfArenas<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics,
    SS: StemStrategy,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaf_extents: Vec::with_capacity(capacity / B + 1),
            leaf_bytes: AVec::new(CACHELINE_ALIGN),
            size: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn new_with_empty_leaf() -> Self {
        unimplemented!("VecOfArenas is immutable-focused and should be constructed from slices")
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let leaf_len = leaf_items.len();

        debug_assert!(leaf_points[0].len() == leaf_len);
        for dim in 1..K {
            debug_assert!(leaf_points[dim].len() == leaf_len);
        }

        let leaf_align = std::mem::align_of::<AX>()
            .max(std::mem::align_of::<T>())
            .max(std::mem::size_of::<u64>());
        let start_offset = align_up(self.leaf_bytes.len(), leaf_align);
        if start_offset != self.leaf_bytes.len() {
            self.leaf_bytes.resize(start_offset, 0u8);
        }
        self.leaf_extents.push((start_offset, leaf_len));

        let mut base = 0usize;
        crate::kd_tree::leaf_view::for_each_leaf_arena_tile_len(leaf_len, |tile_len| {
            for dim in 0..K {
                extend_avec_bytes_from_slice(
                    &mut self.leaf_bytes,
                    &leaf_points[dim][base..base + tile_len],
                );
            }
            extend_avec_bytes_from_slice(&mut self.leaf_bytes, &leaf_items[base..base + tile_len]);

            base += tile_len;
        });

        self.size += leaf_len;
    }
}

#[cfg(test)]
impl<A, T, const K: usize, const B: usize> VecOfArenas<A, T, K, B> {
    pub(crate) fn leaf_bytes_ptr(&self) -> *const u8 {
        self.leaf_bytes.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::VecOfArenas;
    use crate::kd_tree::leaf_strategies::vec_of_arenas::extend_bytes_from_slice;
    use crate::kd_tree::leaf_view::LeafArena;
    use crate::traits_unified_2::{ConstructibleLeafStrategy, LeafStrategy};
    use crate::Eytzinger;

    #[test]
    fn vec_of_arenas_appends_leafs_with_expected_extents() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::new_with_capacity(64);
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let z = [7.0, 8.0, 9.0];
        let items = [10u32, 11, 12];

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        assert_eq!(leaves.size, 3);
        assert_eq!(leaves.leaf_extents.len(), 1);
        assert_eq!(leaves.leaf_extents, vec![(0, 3)]);
        assert_eq!(
            leaves.leaf_bytes.len(),
            LeafArena::<f64, u32, 3>::encoded_len_bytes(3)
        );
    }

    #[test]
    fn vec_of_arenas_decodes_tiled_points_and_items() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::new_with_capacity(64);
        let x: Vec<f64> = (0..40).map(|v| v as f64).collect();
        let y: Vec<f64> = (100..140).map(|v| v as f64).collect();
        let z: Vec<f64> = (200..240).map(|v| v as f64).collect();
        let items: Vec<u32> = (1000..1040).collect();

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        let arena =
            <VecOfArenas<f64, u32, 3, 32> as LeafStrategy<f64, u32, Eytzinger<3>, 3, 32>>::leaf_arena(
                &leaves, 0,
            );

        let mut recovered_x = Vec::new();
        let mut recovered_y = Vec::new();
        let mut recovered_z = Vec::new();
        let mut recovered_items = Vec::new();

        arena.for_each_tiled_chunk(|tile| {
            for idx in 0..tile.len() {
                unsafe {
                    recovered_x.push(tile.point_unaligned(0, idx));
                    recovered_y.push(tile.point_unaligned(1, idx));
                    recovered_z.push(tile.point_unaligned(2, idx));
                    recovered_items.push(tile.item_unaligned(idx));
                }
            }
        });

        assert_eq!(recovered_x, x);
        assert_eq!(recovered_y, y);
        assert_eq!(recovered_z, z);
        assert_eq!(recovered_items, items);
    }

    #[test]
    fn vec_of_arenas_uses_descending_chunk_sizes_per_leaf() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::new_with_capacity(64);
        let x: Vec<f64> = (0..47).map(|v| v as f64).collect();
        let y: Vec<f64> = (100..147).map(|v| v as f64).collect();
        let z: Vec<f64> = (200..247).map(|v| v as f64).collect();
        let items: Vec<u32> = (1000..1047).collect();

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        let arena =
            <VecOfArenas<f64, u32, 3, 32> as LeafStrategy<f64, u32, Eytzinger<3>, 3, 32>>::leaf_arena(
                &leaves, 0,
            );

        let mut tile_lens = Vec::new();
        arena.for_each_tiled_chunk(|tile| tile_lens.push(tile.len()));

        assert_eq!(tile_lens, vec![32, 8, 4, 2, 1]);
    }

    #[test]
    fn extend_bytes_from_slice_appends_exact_repr_bytes() {
        let mut bytes = Vec::new();
        let values = [1u32, 2, 3];

        extend_bytes_from_slice(&mut bytes, &values);

        assert_eq!(bytes.len(), std::mem::size_of_val(&values));
    }
}
