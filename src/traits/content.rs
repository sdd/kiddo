use std::fmt::Debug;

/// Trait required for items stored in the tree.
///
/// This trait needs to be implemented by any type that you want to use to represent the content
/// stored in a [`KdTree`](crate::kd_tree::KdTree).
///
/// Generally this should be [`usize`], [`u32`], or for trees with less
/// than 65,535 points, you could use a [`u16`].
///
/// If you only care about storing the points themselves, you can use [`()`](https://doc.rust-lang.org/std/primitive.unit.html)
/// for the item type.
/// If you do this, you'll need to use a specific constructor for the [`KdTree`](crate::kd_tree::KdTree)
/// ([`KdTree::new_from_slice_no_items`](crate::kd_tree::KdTree::new_from_slice_no_items)).
/// The `best_n_within` query remains available through the fluent query API, but it is not
/// meaningful with `T = ()` because there is no item ordering to define what "best" means.
///
/// Using a [`usize`] is a good start - it's the most ergonomic,
/// since you can use it directly without a cast to index into a [`Vec`] containing more detail
/// on your tree items. If you want to experiment to get better performance or a smaller memory
/// footprint, try switching to a smaller type such as [`u32`].
///
/// Note that you can store non-primitive types for your items if you wish, but performance may
/// suffer with increasing struct size as fewer items fit in the CPU cache.
pub trait Content: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Content for T where T: Copy + Debug + Default + Send + Sync + 'static {}
