# Kiddo

> A high-performance, flexible, ergonomic [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) library.

* [Crate](https://crates.io/crates/kiddo)
* [Documentation](https://docs.rs/kiddo)
* [Usage](#usage)
* [Examples](https://github.com/sdd/kiddo/blob/master/examples/Readme.md)
* [Benchmarks](#benchmarks)
* [Change Log](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md)
* [License](#license)

Kiddo is ideal for super-fast spatial / geospatial lookups and nearest-neighbour / KNN queries for low-ish numbers of dimensions, where you want to ask questions such as:
 - Find the [nearest_n](`float::kdtree::KdTree::nearest_n`) item(s) to a query point, ordered by distance;
 - Find all items [within](`float::kdtree::KdTree::within`) a specified radius of a query point;
 - Find the ["best" n item(s) within](`float::kdtree::KdTree::best_n_within`) a specified distance of a query point, for some definition of "best"

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
kiddo = "2.0.0-beta.5"
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

Criterion is used to perform a series of benchmarks. We compare Kiddo v2 to:
 - Kiddo v1
 - kdtree
 - FNNTW
 - pykdtree


Each action is benchmarked against trees that contain 100, 1,000, 10,000, 100,000, 1,000,000 and in some cases 10,000,000 nodes and charted below.

The `Adding Items` benchmarks are repeated against 2d, 3d and 4d trees. The 3d benchmarks are ran with points that are both of type `f32` and of type `f64`, as well as a 16-bit fixed point use case.

All of the remaining tests are only performed against 3d trees, for expediency. The trees are populated with random source data whose points are all on a unit sphere. This use case is representative of common kd-tree usages in geospatial and astronomical contexts.

The `Nearest n Items` tests query the tree for the nearest 1, 100 and 1,000 points at each tree size. The test for the common case of the nearest one point uses kiddo's `nearest_one()` method, which is an optimised method for this specific common use case.




#### Benchmarking Methodology

*NB*: This section is out-of-date and pertains to kiddo v1. I'll update it soon.

The results and charts below were created via the following process:

* check out the original-kdtree-criterion branch. This branch is the same code as kdtree@0.6.0, with criterion benchmarks added that perform the same operations as the criterion tests in kiddo. For functions that are present in kiddo but not in kdtree, the criterion tests for kdtree contain extra code to post-process the results from kdtree calls to perform the same actions as the new methods in kiddo.

* use the following command to run the criterion benchmarks for kdtree and generate NDJSON encoded test results:

```bash
cargo criterion --message-format json > criterion-kdtree.ndjson
```

* check out the master branch.

* use the following command to run the criterion benchmarks for kiddo and generate NDJSON encoded test results:

```bash
cargo criterion --message-format json --all-features > criterion-kiddo.ndjson
```

* the graphs are generated in python using matplotlib. Ensure you have python installed, as well as the matplotlib and ndjdon python lbraries. Then run the following:

```bash
python ./generate_benchmark_charts.py
```

#### Benchmarking Results

Updated benchmark results will be published soon.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
