use std::fmt::Debug;

/// Trait required for items stored in the tree.
///
/// This trait needs to be implemented by any type that you want to use to represent the content
/// stored in a [`KdTree`](crate::KdTree).
///
/// Generally this will be [`usize`], [`u32`], or for trees with less
/// than 65,535 points, you could use a [`u16`]. The `Basics` blanket implementation covers all of these types.
///
/// If you only care about storing the points themselves, you can use [`()`](https://doc.rust-lang.org/std/primitive.unit.html) for the item type.
/// If you do this, you'll need to use a specific constructor for the [`KdTree`](crate::KdTree)
/// ([`KdTree::new_from_slice_no_items`](crate::KdTree::new_from_slice_no_items)), and the [`best_n_within`](crate::KdTree::best_n_within) query method will be unavailable.
///
/// Using a [`usize`] is a good start - it's the most ergonomic,
/// since you can use it directly without a cast to index into a [`Vec`] containing more detail
/// on your tree items. If you want to experiment to get better performance or a smaller memory footprint,
/// try switching to a smaller type such as [`u32`].
pub trait Basics: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Basics for T where T: Copy + Debug + Default + Send + Sync + 'static {}
