# Kiddo

> A flexible, high-performance kd-tree library.
> Thanks and kudos to mrhooray for the original kdtree library on which kiddo is based.

* [Crate](https://crates.io/crates/kiddo)
* [Documentation](https://docs.rs/kiddo)
* [Usage](#usage)
* [Benchmarks](#benchmarks)
* [License](#license)


## Differences vs Kiddo v1.x

v2 is a complete rewrite from the ground up, with a new internal architecture for much-improved performance. As well as the existing floating point capability, Kiddo v2 now also supports fixed point / integers via the Fixed library. v2 also supports instant zero-copy serialization and deserialization via Rkyv, as well as Serde. See the changelog for a detailed run-down of the changes made since v1.

## Usage
Add `kiddo` to `Cargo.toml`
```toml
[dependencies]
kiddo = "2"
```

Add points to kdtree and query nearest n points with distance function
```rust
use kiddo::KdTree;
use kiddo::distance::squared_euclidean;

let a: ([f64; 2], usize) = ([0f64, 0f64], 0);
let b: ([f64; 2], usize) = ([1f64, 1f64], 1);
let c: ([f64; 2], usize) = ([2f64, 2f64], 2);
let d: ([f64; 2], usize) = ([3f64, 3f64], 3);

let mut kdtree = KdTree::new()?;

kdtree.add(&a.0, a.1)?;
kdtree.add(&b.0, b.1)?;
kdtree.add(&c.0, c.1)?;
kdtree.add(&d.0, d.1)?;

assert_eq!(kdtree.size(), 4);


assert_eq!(
    kdtree.nearest(&a.0, 0, &squared_euclidean).unwrap(),
    vec![]
);
assert_eq!(
    kdtree.nearest(&a.0, 1, &squared_euclidean).unwrap(),
    vec![(0f64, &0)]
);
assert_eq!(
    kdtree.nearest(&a.0, 2, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1)]
);
assert_eq!(
    kdtree.nearest(&a.0, 3, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2)]
);
assert_eq!(
    kdtree.nearest(&a.0, 4, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
);
assert_eq!(
    kdtree.nearest(&a.0, 5, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
);
assert_eq!(
    kdtree.nearest(&b.0, 4, &squared_euclidean).unwrap(),
    vec![(0f64, &1), (2f64, &0), (2f64, &2), (8f64, &3)]
);
```

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




#### Methodology

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

#### Results

The following results were obtained with the above methodology on a machine with these specs:

* AMD Ryzen 5 2500X @ 3600MHz
* 32Gb DDR4 @ 3200MHz

The results are stored inside this repo as `criterion-kiddo.ndjson` and `criterion-kdtree.ndjson`, should you wish
to perform your own analysis.

##### Adding items to the tree
Kiddo generally has a very small performance lead over kdtree@0.6.0 at larger tree sizes, with their performance being similar on smaller trees.

![Charts showing benchmark results for adding items](https://raw.githubusercontent.com/sdd/kiddo/master/benchmark_adding.png)


##### Retrieving the nearest n items

Kiddo's optimised `nearest_one()` method gives a huge performance advantage for single item queries, with up to 9x faster performance.
Kiddo's standard `nearest()` method also outperforms kdtree@0.6.0.

![Charts showing benchmark results for retrieving the nearest n items](https://raw.githubusercontent.com/sdd/kiddo/master/benchmark_nearest_n.png)

##### Retrieving all items within a distance, sorted
Things look closer here at first glance but the logarithmic nature of the charted data may obscure the fact that Kiddo is often up to twice as fast as kdtree@0.6.0 here.

![Charts showing benchmark results for retrieving all items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo/master/benchmark_within.png)

##### Retrieving all items within a distance, unsorted
kdtree@0.6.0 does not have a `within_unsorted()` method, so we are comparing kiddo's `within_unsorted()` to kdtree@0.6.0's `within()` here, with kiddo up to 5x faster on the million-item tree.

![Charts showing benchmark results for retrieving all items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo/master/benchmark_within_unsorted.png)

##### Retrieving the best n items within a specified distance
Kiddo's performance advantage here ranges from twice as fast for hundred-item trees up to as much as 20x faster for trees with a million items.

![Charts showing benchmark results for retrieving the best n items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo/master/benchmark_best_n_within.png)

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
