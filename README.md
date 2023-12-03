# Kiddo

> A high-performance, flexible, ergonomic [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) library. Possibly the fastest k-d tree library in the world? [See for yourself](https://sdd.github.io/kd-tree-comparison-webapp/).

* [Crate](https://crates.io/crates/kiddo)
* [Documentation](https://docs.rs/kiddo)
* [Usage](#usage)
* [Examples](https://github.com/sdd/kiddo/blob/master/examples/Readme.md)
* [Benchmarks](#benchmarks)
* [Change Log](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md)
* [License](#license)

Kiddo is ideal for super-fast spatial / geospatial lookups and nearest-neighbour / KNN queries for low-ish numbers of dimensions, where you want to ask questions such as:
 - Find the [nearest_n](https://docs.rs/kiddo/latest/kiddo/float/kdtree/struct.KdTree.html#method.nearest_n) item(s) to a query point, ordered by distance;
 - Find all items [within](https://docs.rs/kiddo/latest/kiddo/float/kdtree/struct.KdTree.html#method.within) a specified radius of a query point;
 - Find the ["best" n item(s) within](https://docs.rs/kiddo/latest/kiddo/float/kdtree/struct.KdTree.html#method.best_n_within) a specified distance of a query point, for some definition of "best".

Kiddo provides:
 - Its standard floating point k-d tree, exposed as [`kiddo::KdTree`](`crate::KdTree`)
 - **integer / fixed point support** via the [`Fixed`](https://docs.rs/fixed/latest/fixed/) library;
 - **instant zero-copy deserialization** and serialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).
 - An [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) with space and performance advantages over the standard
   k-d tree, for situations where the tree does not need to be modified after creation

## Usage
Add `kiddo` to `Cargo.toml`
```toml
[dependencies]
kiddo = "3.0.0"
```

Add points to kdtree and query nearest n points with distance function
```rust
use kiddo::{KdTree, SquaredEuclidean};

let entries = vec![
    [0f64, 0f64],
    [1f64, 1f64],
    [2f64, 2f64],
    [3f64, 3f64]
];

// use the kiddo::KdTree type to get up and running quickly with default settings
let mut tree: KdTree<_, 2> = (&entries).into();

// How many items are in tree?
assert_eq!(tree.size(), 4);

// find the nearest item to [0f64, 0f64].
// returns an instance of kiddo::NearestNeighbour
let nearest = tree.nearest_one::<SquaredEuclidean>(&[0f64, 0f64]);
assert_eq!(nearest.distance, 0f64);
assert_eq!(nearest.item, 0);


// find the nearest 3 items to [0f64, 0f64]
// // returns an Vec of kiddo::NearestNeighbour
let nearest_n: Vec<_> = tree.nearest_n::<SquaredEuclidean>(&[0f64, 0f64], 3);
assert_eq!(
    nearest_n.iter().map(|x|(x.distance, x.item)).collect::<Vec<_>>(),
    vec![(0f64, 0), (2f64, 1), (8f64, 2)]
);
```
See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more detailed examples.

## Optional Features

The Kiddo crate exposes the following features. Any labelled as **(NIGHTLY)** are not available on `stable` Rust as they require some unstable features. You'll need to build with `nightly` in order to user them.
* `serialize` - serialization / deserialization via [`Serde`](https://docs.rs/serde/latest/serde/)
* `serialize_rkyv` - zero-copy serialization / deserialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/)
* `global_allocate` **(NIGHTLY)** -  When enabled Kiddo will use the unstable allocator_api feature within [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) to get a slight performance improvement when allocating space for leaves.
* `simd` **(NIGHTLY)** - enables some hand-written SIMD intrinsic code within [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) that may improve performance (currently only on the nearest_one method when using `f64`)

## v3.x

Version 3.x changed the distance metrics syntax, switching from function pointers to a trait-based
approach that permitted some ergonomics and performance improvements. This is a breaking change though:
whereas prior to v3, you may have had queries that look like this:

```
use kiddo::distance::squared_euclidean;
let result = kdtree.nearest_one(&[0f64, 0f64], &squared_euclidean);
```

Now in v3, you'll need to switch to this syntax:

```
use kiddo::SquaredEuclidean;
let result = kdtree.nearest_one::<SquaredEuclidean>(&[0f64, 0f64]);
```

V3 also introduces the [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) variant. Designed for use cases where all the points that you need to add
to the tree are known up-front, and no modifications need to be made after the tree is initially populated.
[`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) balances and optimises the tree at construction time, ensuring much more efficient
memory usage (and a correspondingly smaller size on-disk for serialized trees). Since the interior
nodes of the [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) also take up less space in memory, more of them can fit in the CPU cache, potentially
improving performance in some cases.

## v2.x

Version 2.x was a complete rewrite, providing:
- a new internal architecture for **much-improved performance**;
- Added **integer / fixed point support** via the [`Fixed`](https://docs.rs/fixed/latest/fixed/) library;
- **instant zero-copy deserialization** and serialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).
- See the [changelog](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md) for a detailed run-down on all the changes made in v2.


## Benchmarks

The results of all the below benchmarks are viewable in an interactive webapp over at [https://sdd.github.io/kd-tree-comparison-webapp/](https://sdd.github.io/kd-tree-comparison-webapp/).

The comparative benchmark suite is located in another project, [https://github.com/sdd/kd-tree-comparison](https://github.com/sdd/kd-tree-comparison).

Criterion was used to perform a series of benchmarks. We compare Kiddo v3 to:
* Kiddo v2.x
* [Kiddo v1.x / v0.2.x](https://github.com/sdd/kiddo_v1)
* [FNNTW](https://crates.io/crates/fnntw) v0.2.3
* [nabo-rs](https://crates.io/crates/nabo) v0.2.1
* [pykdtree](https://github.com/storpipfugl/pykdtree) v1.3.4
* [sklearn.neighbours.KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) v1.2.2
* [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) v1.10.1

The following activities are benchmarked (where implemented):
* Construction of a k-d tree from a list of points and indexes
* Querying the nearest one, ten, or one hundred points to a given query point
* Querying all points within a set radius of a given point (both unsorted results, and results sorted by distance)
* Querying the nearest n items within a specified radius (sorted and unsorted)
* 
Each action is benchmarked against trees that contain 100, 1,000, 10,000, 100,000, 1,000,000 and in some cases 10,000,000 nodes.

The benchmarks are repeated against 2d, 3d and 4d trees, as well as with points that are both of type `f32` and of type `f64`, as well as a 16-bit fixed point use case for Kiddo v2.

The trees are populated with random source data whose points are all on a unit sphere. This use case is representative of common kd-tree usages in geospatial and astronomical contexts.


## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
