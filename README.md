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

## Differences vs Kiddo v1.x

Version 2.x is a complete rewrite, providing:
- a new internal architecture for **much-improved performance**;
- Added **integer / fixed point support** via the [`Fixed`](https://docs.rs/fixed/latest/fixed/) library;
- **instant zero-copy deserialization** and serialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).
- See the [changelog](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md) for a detailed run-down on all the changes made in v2.

## Usage
Add `kiddo` to `Cargo.toml`
```toml
[dependencies]
kiddo = "3.0.0-beta.1"
```

Add points to kdtree and query nearest n points with distance function
```rust
use kiddo::KdTree;
use kiddo::distance::squared_euclidean;

let entries = vec![
    [0f64, 0f64],
    [1f64, 1f64],
    [2f64, 2f64],
    [3f64, 3f64]
];

// use the kiddo::KdTree type to get up and running quickly with default settings
let mut kdtree: KdTree<_, 2> = (&entries).into();

// How many items are in tree?
assert_eq!(kdtree.size(), 4);

// find the nearest item to [0f64, 0f64].
// returns a tuple of (dist, index)
assert_eq!(
    kdtree.nearest_one(&[0f64, 0f64], &squared_euclidean),
    (0f64, 0)
);

// find the nearest 3 items to [0f64, 0f64], and collect into a `Vec`
assert_eq!(
    kdtree.nearest_n(&[0f64, 0f64], 3, &squared_euclidean).collect::<Vec<_>>(),
    vec![(0f64, 0), (2f64, 1), (8f64, 2)]
);
```
See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more detailed examples.

## Benchmarks

The results of all the below benchmarks are viewable in an interactive webapp over at [https://sdd.github.io/kd-tree-comparison-webapp/](https://sdd.github.io/kd-tree-comparison-webapp/).

The comparative benchmark suite is located in another project, [https://github.com/sdd/kd-tree-comparison](https://github.com/sdd/kd-tree-comparison).

Criterion was used to perform a series of benchmarks. We compare Kiddo v2 to:
* [Kiddo v1.x / v0.2.x](https://github.com/sdd/kiddo_v1)
* [FNNTW](https://crates.io/crates/fnntw) v0.2.3
* [nabo-rs](https://crates.io/crates/nabo) v0.2.1
* [pykdtree](https://github.com/storpipfugl/pykdtree) v1.3.4
* [sklearn.neighbours.KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) v1.2.2
* [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) v1.10.1

The following activities are benchmarked:
* Construction of a k-d tree from a list of points and indexes
* Querying the nearest one, ten, or one hundred points to a given query point
* Querying all points within a set radius of a given point (both unsorted results, and results sorted by distance)

Each action is benchmarked against trees that contain 100, 1,000, 10,000, 100,000, 1,000,000 and in some cases 10,000,000 nodes.

The benchmarks are repeated against 2d, 3d and 4d trees, as well as with points that are both of type `f32` and of type `f64`, as well as a 16-bit fixed point use case for Kiddo v2.

The trees are populated with random source data whose points are all on a unit sphere. This use case is representative of common kd-tree usages in geospatial and astronomical contexts.


## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
