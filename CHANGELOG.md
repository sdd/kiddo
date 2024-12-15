# Kiddo Changelog

## [6.0.0] - 2024-12-15

### Chore

- Rename from sok (son of kiddo) to kiddo, with the aim of this being kiddo v2. Add some docs
- Update some docs and add beta suffix to version
- Bump version to 2.0.0-beta.5
- Update .gitignore
- Bump version to 2.0.0-beta.8
- Bump version to 2.0.0-beta.9
- Update actions/checkout action to v3
- Bump version to 2.0.2
- Add .DS_Store to .gitignore
- Add some missing docs and hide some private internals
- Update rust crate criterion to 0.5
- Fix clippy lints
- Update versions in lib.rs
- Release
- Fix clippy lints
- Add .codecov.yml
- Bump version to 2.1.2
- Update actions/checkout action to v4
- Update ad-m/github-push-action action to v0.8.0
- Remove commented-out code in float's within_into_iter
- Enable some optimizations on the bench profile
- Release 3.0.0-beta.1
- Add DS_Store and vscode to .gitignore
- Update trace calls
- Minor immutable tweaks
- Minor example and docs update for immutable
- Merged in changes from 3.0 branch
- Release 3.0.0-beta.3
- Remove commented code and test comments
- Version 3.0.0-beta.4
- Reorg simd conditional compilation annotations
- Fix clippy lints
- Remove unused deps
- Update ordered-float dep from 3.7 -> 4
- Version 3.0.0-rc.1
- Update actions/checkout action to v4
- Downgrade serde_with to fix failing dependency resolution
- Clean up some clippy lints from tests
- V3.0.0 :tada:
- Consistency
- Update rust crate proc-macro2 to 1.0.70
- Remove an example file as it is causing problems with release-plz 🙄
- Release, Signed-off-by:Scott Donnelly <scott@donnel.ly>
- Remove unused import
- Update actions/cache action to v4
- Update codecov/codecov-action action to v4
- Update baptiste0928/cargo-install action to v3
- Release
- Release
- Release, Signed-off-by:Scott Donnelly <scott@donnel.ly>
- Refactor trait bounds to silence new clippy lints
- Silence new lint error
- Silence error regarding doc_cfg no longer working
- Update codspeedhq/action action to v3
- Release
- Fix some lint issues
- Remove unused code
- Update codecov/codecov-action action to v5
- Update colathro/crate-version action to v2
- Release
- Release v5.0.2
- Update version references
- Update version references

### Ci

- Add build workflow
- Add CI for format/clippy/test/coverage/release/publish
- Fix git-cliff.toml changelog config
- Update pre-release.yml to use a single job
- Update pre-release.yml to remove Clippy check
- Update release.yml to only release on merged PRs labelled `release`
- Add job to update version strings in docs on pre-release PR branch
- Temp disable testing of SIMD code path
- Ensure doctests get ran in CI
- Add codspeed
- Ensure codspeed runs on master rather than main. Don't show progress on CI checkout
- Fix issue with CI lint steps
- Restore accidentally removed step from pre release string udpater job
- Fix pre-release string updater regex

### Deps

- Relax strictness of some deps to reduce renovate noise

### Revert

- Changes since 'next' branch diverged
- Re-apply changes reverted in 42f9455

### ♻️ Refactor

- Move Content and Index to src/types.rs and use for both float and fixed. Refactor some repeated test code into test_utils
- Reduce boilerplate in nearest_one and nearest_n benches
- Tighten unsafe boundaries. Rename best_n_within_into_iter to best_n_within
- Dont store bounds on nodes
- Remove unstable features to allow use on stable rust
- Update `Axis` trait to include some methods so that the
- Float and fixed both use a common macro for nearest_n
- Float and fixed both use a common macro for within
- Float and fixed both use a common macro for within_unsorted
- Float and fixed both use a common macro for best_n_within
- Use idx 1 as root. return usize from size()
- Use usize where poss to index stem nodes in add and split
- Rewrite add and split to improve performance
- Pass leaf node count to stem optimizer
- Extract extend_shifts function
- Clippy lint fixes
- Remove unused code from simd leaf node
- Slight tweak to SortedVec NearestNeighbour max_dist
- Remove need for gated import
- Remove unstable features
- Switch from having an eytzinger feature to a modified_van_emde_boas feature and default to Eytzinger. Update readme and changelog
- Split LeafSliceFloatChunk out of LeafSliceFloat to fix unspecifiable parameter issue. Fix some lint issues and tests, and update docs
- Unify types and distance_metric modules into a single unified traits modules to make things a bit neater, especially in the docs
- Modified_van_emde_boas feature gating rewritten. Separate tests for branchless and non-branchless mveb. Fix some new lints

### ⚡️ Performance

- Refactor `within` to simply sort the result of `within_unsorted`
- Quit as early as possible if chink fits in bucket
- Revert perf regression on immutable tree. Update docs

### ✨ Features

- Initial commit
- Add rkyv zero-copy deserialization
- Added SIMD f32 4D specific tree. Lots of perf improvements
- Refactor simd to tuned. add benches for tuned f32d4 and u16d4. refactor leafnodeentry into separate content points and items for better autovectorization
- Add generic fixed point tree
- Remove saturating mul / add from fxp squared_euclidean to try to make it vectorizible. split leaf processing in best_n_into_iter into separate funcs to track runtime of each part better
- Refactor into float and fixed. Fixed mirror_select_nth_unstable_by bug. Add nearest_n
- Add remove() method
- Add Sync bound to Axis, Content and Index to allow use with Rayon
- Bump version to 2.0.0-beta.2
- Update deps, add serde and rkyv examples, clean up custom serde, remove main
- Add changelog and example links to readme. Update version strings
- Add default tree export
- Return Neighbour<> instead of a tuple in queries., BREAKING:nearest_n now returns a Vec<Neighbour<_, _>> instead of an iterator.
  The MinMaxHeap was removed in favour of a BinaryHeap, which pretty much doubled
  query performance on nearest_n but it means that returning a Vec<> here instead
  of an Iterator makes more sense as we need to call to_sorted_vec() on the
  BinaryHeap anyway, so we may as well just return the resulting Vec rather than
  converting that to an Iterator since the most common use case would be to then
  collect that Iterator into a Vec anyway
- Rename `radius` param to `dist` and update documentation
- V2.0.0 release
- Implement all queries on float::ArchivedKdTree. Update rkyv example.
- Queries return structs instead of tuples
- Use a trait instead of a function pointer for distance metrics
- Add within_unsorted_into_iter query
- Semi-static-stem
- Add nearest_one implementation
- Balanced construction
- Balanced construction part 2
- Balanced construction pt 3
- Add immutable kdtree
- Use shifts array during construction, and other fixes and optimizations
- It's finally working! Tested on 1m random trees of each size from 16 to 100, no errors. Tests include an 8m item tree
- Immutable WIP
- Immutable WIP 2 (all queries done)
- Gate unstable rust features behind a crate feature
- Rename ImmutableTree::optimize_from to new_from_slice. Update docs and version to 3.0.0-beta.2
- ImmutableTree uses Best/NearestNeighbour for query results, and uses DistanceMetric
- Refactor immutable to macros to allow DRY query method generation for ArchivedImmutableKdTree
- Implement query methods on ArchivedImmutableKdTree using macros
- Simd leaf
- SIMD leaf nearest one
- SIMD leaves, AVX2 f64
- SIMD leaf nearest one: f32 and avx2
- SIMD leaf nearest n within
- Export float ImmutableKdTree and Manhattan from root. Clean up examples.
- Add immutable best_n_within
- `ImmutableKdTree` now works on stable, BREAKING CHANGE:the `immutable` crate feature now no longer exists.
- Make tracing an optional dependency gated by tracing feature flag
- Iterate over trees
- Make rand and rayon optional
- Add f16 support, example and docs to show usage with half crate
- Make nearest_n_within max_qty param a NonZero
- Rename some features to be more conventional and fix the way they work
- Experimental ImmutableTree variant with dynamic split dimension
- Pointcloud test examples
- Modified van Emde Boas layout
- Flesh out immutable dynamic variant
- More immutable dynamic enhancements
- Implement remaining immutable_dynamic query methds
- Replace immutable with immutable dynamic
- Ensure immutable works when empty or single bucket. Fix broken tests
- Use avec to force alignment for up to 40% query perf improvement
- Add eytzinger feature to test perf vs mod. van emde boas

### 🐛 Bug Fixes

- Allow creation with capacity zero, Issue:https://github.com/sdd/kiddo/issues/11
- Bug in remove when query point value has same value as a split plane, Issue:https://github.com/sdd/kiddo/issues/12
- Update rust crate fixed to 1.23
- Update rust crate rayon to 1.7
- Properly split buckets, Fixes:https://github.com/sdd/kiddo/issues/28
- Update rust crate serde_with to v3
- Incompatibility with fixed crate num-traits feature
- Update rust crate rayon to 1.8
- Prevent occasional overflow in `Fixed` tree queries by using a saturating_add
- Only return from root of stem optimizer if there is no room in existing leaves
- Only terminate stem optimizer early if the upper child fits in one bucket as well as the lower
- Stem optimizer passes capacity of subtree rather than leaf node count. Fails on 1 tree in 16m for sizes 16-32
- Optimize_Stems handles right subtree shift requests properly
- Tracking down last immutable bugs
- Immutable pivot calc improved. Tracing. bupe checking test
- One more immutable bug
- Update conditional compilation directives and gate unstable lang features
- Disable AVX512 until I can test it on a machine with actual AVX512
- Don't use AVX512 intrinsic in AVX2 code path
- Update rust crate serde_with to 3.4
- Simd in immutable::nearest_one
- AVX512 missing param
- Update rust crate itertools to 0.12
- Re-enable support for wasm targets
- Stdsimd removed from unstable, fix:fixup simd removal
- Add missing global_allocate feature definition and sort feature defs alphabetically
- Update benches to require test_utils feature. update clippy and test steps to include new test_utils feature
- Claytonwramsey bug, Fixes:[#138](https://github.com/sdd/kiddo/pull/138)
- Prevent overflow in capacity_with_bucket_size on non-64 bit architectures
- Prevent assertion failure when stem optimisation needs a large shift
- Nearest_n_within does not limit num of items when not sorted, Issue:https://github.com/sdd/kiddo/issues/168
- Update rust crate itertools to 0.13
- Avx2-check example fails on rust 1.81
- Get rkyv ser/de working for ImmutableKdTree. Update examples and docs
- Consistently use CACHELINE_ALIGN
- Only export WithinUnsortedIter on archs where the within_unsorted_iter mod is configured
- Broken release action yaml
- Repro test failure from pre-release job in build job
- Disable broken get_best_from_dists_f64_avx2 until fixed
- Pre-release string updater job branch matcher
- Syntax error from pre-release string updater github action bug

### 💄 Styling

- Fmt
- Fmt
- Spelling fix and comment removal
- Fmt

### 📝 Documentation

- Use only 5 keywords in Cargo.toml
- Cargo.toml keywords can't contain spaces
- Remove reference to kiddo being based on kdtree now that this is a rewrite
- Add documentation and doc examples
- Add documentation for add/remove methods
- Add cities example
- Fix some broken links and add missing docs for last generic param in cities example
- Update changelog and docs for release 2.1.0
- Fix serde example
- Minor documentation enhancements
- Update rkyv-large example
- Update version in README.md
- Add workings for right subchild example
- Fix bad links
- Minor documentation tweaks
- Clean up some examples
- Rewrite doctests to use convenient top-level exports. Ensure doctest .rkyv artifact is reproducible
- Fix ImmutableKdTree links in the top level documentation
- Update example for ImmutableKdTree::size
- Update feature docs in lib.rs
- Update changelog
- Update references of kd-tree to k-d tree. Minor rewrite of other parts
- More documentation improvements
- Update changelog

### 🧪 Testing

- Add float construction many items unit test
- Update benches
- Refactor add and nearest_one benchmarks to be more DRY
- Refactor all benches to be more DRY. Format and fix some clippy lints
- Fix nearest_n bench code
- Re-enable serde test
- Add tests to cover all split conditions
- Add bench for nearest_one
- Optimize stems stress test only goes to size 32 but seed 1m
- Fix doctests
- Sort result of within() call in tests to prevent spurious test failures
- Fix iter doctests and remove unused var
- Add hacky workaround to enable tests to run without having to specify --features=test_utils
- Remove unneeded large test and ignored tests
- Add tests for float slice fallback

## [5.0.2] - 2024-12-15

### 🐛 Bug Fixes

- Disable broken get_best_from_dists_f64_avx2 until fixed
- only export WithinUnsortedIter on archs where the within_unsorted_iter mod is configured (https://github.com/sdd/kiddo/issues/190)
- fix: consistently use CACHELINE_ALIGN (https://github.com/sdd/kiddo/issues/190)

## [5.0.1] - 2024-12-08

### Performance

- fix a performance regression on the immutable tree.

### Documentation

- ensure that the top-level documentation example shows v5 being imported.


## [5.0.0] - 2024-11-30

Version 5 bundles a complete re-write of [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) alongside some rationalization of feature names and a change of type of the `max_qty` parameter present in some query methods from `usize` to `NonZero<usize>`.

### `ImmutableKdTree` rewrite

**BREAKING CHANGE**: For anyone that has been serializing `ImmutableKdTree` (using either `serde` or `rkyv`), version 5 constitutes a breaking change as serialized trees from prior versions will not be deserializable with v5 and vice-versa.

Quite a few people (https://github.com/sdd/kiddo/issues/172, https://github.com/sdd/kiddo/issues/158, https://github.com/sdd/kiddo/issues/78) have previously unsuccessfully tried to use `ImmutableKdTree` with data containing many points that have the same value on one or more of their axes, for example point cloud data containing many points on a flat axis-aligned plane.
The v5 rewrite of `ImmutableKdTree` experiences none of these kinds of problems and can be safely used no matter what your data looks like.
Query performance is in many cases faster than the prior version, but sometimes slightly slower - your mileage may vary but differences in query performance is pretty small.
Construction performance is considerably improved, with up to a 2x speedup, with the improvement becoming more pronounced as the tree size increases.
Memory efficiency is slightly better also.

Behind-the-scenes, the structure of the `ImmutableKdTree` has changed from using a Vec of fixed-size array-based buckets to using a single array-of-vecs to store all the points, with per-bucket offsets being stored for each leaf. To avoid dynamic allocation at query time, a fixed slice that chunks the bucket is used, permitting autovectorisation to work well and giving the opportunity for manual SIMD to be used on the fixed-length slice. Trailing values beyond the last full slice are processed individually.

### Modified van Emde Boas Stem Ordering

The experimental `modified_van_emde_boas` feature allows an alternative stem node ordering mode to be enabled.
When enabled, the ordering of stem nodes changes from using Eytzinger ordering to a modified van Emde Boas (can I call this a Donnelly ordering? :-p) order.
This is a novel implementation unique to Kiddo v5 that ensures that a cache line only needs to be retrieved at most once every three levels (on most CPUs when using f64), or every four levels (on most CPUs when using f32). This increases by an extra one level on CPU architectures with a 128-byte cache line width (this is quite rare at the moment but can be found on some Apple M3 and newer CPUs).
Previous literature has indicated that a standard van Emde Boas layout provided no advantage, but thanks to an efficient branchless implementation of the stem ordering logic, and a refinement to leave the last slot on each cache line empty, rather than straddling levels across cache lines, cache efficiency is improved to the point where gains can sometimes be seen over the previously-best Eytzinger layout.
Typically, performance varies from between 1% faster an 5% slower than Eytzinger, from what I've seen during testing, with the differences often being statistically insignificant.

### `ImmutableKdTree` + `rkyv`

The v5 `ImmutableKdTree` uses an Aligned Vec internally for storing stem nodes. It is not possible to zero-copy deserialize
into an Aligned Vec with `rkyv` as there is no guarantee that the stem vec in the underlying buffer respects the alignment.
As such, unfortunately this means that `ImmutableKdTree` itself can't be fully zero-copy serialized / deserialized, but there
are some related types that are provided that allow zero-copy deserialization to be performed for all other parts of the tree
except for the stems, which themselves get copied into an aligned array from the buffer.
In practice this is still very fast as the stems are only a very small part of the overall tree.

See `immutable-rkyv-serialize` and `immutable-rkyv-deserialize` in the examples for how to do this.

### Feature name changes

**BREAKING CHANGE**: It was pointed out in https://github.com/sdd/kiddo/issues/159 that it was necessary to enable both `rkyv` and `serialize_rkyv` features to use Rkyv serialization. I took the opportunity of the major version bump to rationalize the feature names to make them easier to use.
`serialize_rkyv` has been removed and now only `rkyv` feature is needed to enable Rkyv serialization.
`serialize` has been renamed to `serde` in line with ecosystem conventions.
`half` has been renamed to `f16` for clarity.

### `max_qty` Changed to `NonZero<usize>`

**BREAKING CHANGE**: It was noted by [@ezrasingh](https://github.com/sdd/kiddo/issues/168#issuecomment-2335183999) that specifying `max_qty` as `0` in version 4.2.1 alongside `sorted = false` resulted in a panic. Since requesting a `max_qty` of zero makes no sense, and to avoid adding a run-time check for users who have no possibility of specifying a `max_qty` of `0`, the type of `max_qty` has been changed to `NonZero<usize>` to make this a compile-time check instead.


## [4.2.1] - 2024-08-17

### Chore

- Refactor trait bounds to silence new clippy lints
- Silence new lint error
- Silence error regarding doc_cfg no longer working
- Update codspeedhq/action action to v3
- Update rust crate generator to 0.8.1

### 🐛 Bug Fixes

- Nearest_n_within does not limit num of items when not sorted, Issue:https://github.com/sdd/kiddo/issues/168
- Update rust crate itertools to 0.13

## [4.2.0] - 2024-02-18

### ✨ Features

- Add f16 support, example and docs to show usage with half crate

### 🐛 Bug Fixes

- Prevent assertion failure when stem optimisation needs a large shift

## [4.1.1] - 2024-02-17

### 🐛 Bug Fixes

- Prevent overflow in capacity_with_bucket_size on non-64 bit architectures

## [4.1.0] - 2024-02-17

### Chore

- Remove unused import
- Update actions/cache action to v4
- Update codecov/codecov-action action to v4
- Update baptiste0928/cargo-install action to v3

### Ci

- Fix issue with CI lint steps

### Deps

- Relax strictness of some deps to reduce renovate noise

### ♻️ Refactor

- Remove need for gated import

### ✨ Features

- Make tracing an optional dependency gated by tracing feature flag
- Iterate over trees
- Make rand and rayon optional

### 🐛 Bug Fixes

- Stdsimd removed from unstable, fix:fixup simd removal
- Add missing global_allocate feature definition and sort feature defs alphabetically
- Update benches to require test_utils feature. update clippy and test steps to include new test_utils feature
- Claytonwramsey bug, Fixes:[#138](https://github.com/sdd/kiddo/pull/138)

### 🧪 Testing

- Fix iter doctests and remove unused var
- Add hacky workaround to enable tests to run without having to specify --features=test_utils

## [4.0.0] - 2023-12-04

Despite the major version bump, this is unlikely to be a breaking change for any users. The `within_unsorted_iter` method of `ImmutableKdTree` is now only present on x86_64 and Aarch64 targets.
Considering that v3.0.0 would not even compile on these targets when the `immutable` crate feature was activated, 
it seems vanishingly unlikely that this breaks anyone.
Additionally, the `immutable` feature has been removed and the `global_allocate` feature added. If you were using `ImmutableKdTree` and your build
breaks because the `immutable` feature does not exist - don't worry, you don't need it anymore.
Simply remove any reference to it ant the `ImmutableKdTree` should be available without it.

### ✨ Features

- `ImmutableKdTree` now works on stable

### 🐛 Bug Fixes

- Update rust crate itertools to 0.12
- Re-enable support for wasm targets

### 📝 Documentation

- Update feature docs in lib.rs


## [3.0.0] - 2023-11-05

I can't believe how long it has taken me to get v3 into shape, but it's finally here! :tada:

The [ImmutableKdTree](https://docs.rs/kiddo/3.0.0/kiddo/immutable/float/kdtree/struct.ImmutableKdTree.html) is finally ready! :tada: Designed for use cases where all the points that you need to add
to the tree are known up-front, and no modifications need to be made after the tree is initially populated.
`ImmutableKdTree` balances and optimises the tree at construction time, ensuring much more efficient
memory usage (and a correspondingly smaller size on-disk for serialized trees). Since the interior
nodes of the `ImmutableKdTree` also take up less space in memory, more of them can fit in the CPU cache, potentially
improving performance in some cases.

The `immutable` crate feature needs to be activated in order to use `ImmutableKdTree`.
More info on `ImmutableKdTree` can be found below in the 3.0.0 beta and RC changelog entries.

Version 3.x changes the distance metrics syntax, switching from function pointers to a trait-based
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

## [3.0.0-rc.1] - 2023-10-17

### Features
* the `ImmutableKdTree` is now only usable by enabling the `immutable` crate feature. This ensures that the crate as a whole retains compatible with stable rust, as `ImmutableKdTree` depends on some unstable features at present.

### Refactors
* Leaf nodes for Immutable now store their points in columnar format. Searches across them have been re-written to autovectorise better. This has been tested on Compiler Explorer to demonstrate that AVX512 instructions are generated, ensuring vectorization is as wide as is possible. Handwritten SIMD intrinsics have been used (activated by enabling the `simd` crate feature) to manually vectorise code that the compiler could not autovectorise. **NOTE** `simd` is currently quite unstable and not as well tested as the rest of the library, so use it with caution until it stabilizes in the full `v3.0.0` release!


### Style / Tests
* Increase reliability of `within()` test for `ImmutableKdTree`.
* Remove some commented-out code and some useless comments



## [2.1.2] - 2023-10-10

### Fixes

- fix incompatibility with the `num-traits` feature of the `fixed` crate


## [3.0.0-beta.4] - 2023-08-28

### Style / Tests
* Increase reliability of `within()` test for `ImmutableKdTree`.
* Remove some commented-out code and some useless comments in some tests

## [3.0.0-beta.3] - 2023-08-28

### Features
* Implement query methods on `ArchivedImmutableKdTree`, similarly to `ArchivedKdTree`

## [3.0.0-beta.2] - 2023-08-26

Introducing the [ImmutableKdTree](https://docs.rs/kiddo/3.0.0-beta.2/kiddo/immutable/float/kdtree/struct.ImmutableKdTree.html) for floating point! :tada:

`ImmutableKdTree` is intended for use when the smallest possible on-disk serialized size of a tree is of paramount importance, and / or the fastest possible query speed is required.

Expect improvements in query time of 10-15%, and a reduction in the size of serialized trees by 33% or so on average.

These capabilities come with a few trade-offs:
1) This tree does not provide the capability to modify its contents after it has been constructed. The co-ordinates of the points to be stored must have all been generated in advance.
2) Construction time can be quite a bit slower. Typically, this can be twice as long as the default `kiddo::float::kdtree::KdTree`.
3) The more common that duplicate values are amongst your source points, the slower it will take to construct the tree. If you're using `f64` data that is fairly random-ish, you will probably not encounter any issues. I've successfully created 250 million node `ImmutableTree` instances with random `f64` data with no issues, limited only by RAM during construction. Likewise for `f32` based trees, up to a few million nodes. As per the other Kiddo float-type trees, points being stored in the tree must be floats (`f64` or `f32` are supported currently).

## [3.0.0-beta.1] - 2023-06-18

### Breaking Changes

* feat!: queries return structs instead of tuples. Query methods have been updated so that they all return
  either a `NearestNeighbour`, `Vec<NearestNeighbour>`,
  or `Vec<BestNeighbour>`, for consistency.
* feat!: use a trait instead of a function pointer for distance metrics (See [SquaredEuclidean](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/distance/struct.SquaredEuclidean.html) and [Manhattan](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/distance/struct.Manhattan.html))
* feat: add [within_unsorted_iter](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within_unsorted_iter) query

### Performance

* perf: refactor [within](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within) to simply sort the result of [within_unsorted](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within_unsorted).
  Previously, `within` was keeping its results in a `BinaryHeap` and calling
  its `into_sorted_vec` method to, well, return a sorted `Vec`.
  Whilst a `BinaryHeap` is great if you are frequently adding and removing
  items, if your use case is to gradually add all your items, and then sort
  them all at once, it's quicker to just put things in a `Vec` and then
  sort the `Vec` at the end.
  Benchmarking shows that this change improves performance by anything from
  5 to 60% in practice.


## [2.1.1] - 2023-06-07

### Refactor

- update Axis trait to include some methods so that the `nearest_one` methods can be identical between `float` and `fixed`.
- float and fixed both use a common macro for best_n_within
- float and fixed both use a common macro for within_unsorted
- float and fixed both use a common macro for within
- float and fixed both use a common macro for nearest_n

### CI

- Update pre-release.yml to remove Clippy check
- Add CI for format/clippy/test/coverage/release/publish

## [2.1.0]
* feat: implement the main query methods plus `size` on `kiddo::float::kdtree::ArchivedKdTree` and improve the Rkyv example.

The previous Rkyv example was not really using Rkyv in the most efficient way (Thanks to @cavemanloverboy for spotting my mistakes!). In order to properly use rkyv's zero-copy deserialization, you need to use `rkyv::archived_root` to transmute a buffer into an `ArchivedKdTree`. For `ArchivedKdTree` to be useful, it actually needs some methods though!

v2.1.0 refactors the query code so that the method bodies of the queries are templated by macros, allowing them to be implemented on `KdTree` and `ArchivedKdTree` without completely duplicating the code.

The updated rkyv example shows the difference that zero-copy usage of rkyv makes vs deserializing, as well as also showing the gains that can be made using mmap compared to standard file access. Combining both together results in absolutely mind-blowing performance when measuring  time-from-binary-start-to-first-query-result.

See for yourself by downloading the sample datasets mentioned in the examples readme and running:

```sh
cargo run --example rkyv --features=serialize_rkyv --release
```

On my machine, using the old technique of normal file access and deserialization into `KdTree`, the example code takes 348 milliseconds to load and query. The memmapped code that just transmutes to an `ArchivedKdTree` and then queries it takes 182 **micro** seconds(!) - an improvement by a factor of 1900x!!

I'll follow up this release with equivalent methods for `Fixed`, and some more ergonomic methods for loading and saving.

## [2.0.2]
* fix: properly split buckets.
  Previously, when a bucket had multiple items with the same value in the splitting dimension as the split plane, and these values straddled the pivot point, some items could end up in the wrong bucket after the split.

## [2.0.1]
 * refactor: removed the requirement to use unstable features so that Kiddo should now work on Rust stable.

## [2.0.0]

Version 2 is a complete rewrite and re-architecting of Kiddo. The same methods have been provided (except periodic boundary conditions, for now), but large performance improvements have been made across the board, and some improvements have been made to the ergonomics of the library also.
Needless to say, this is a breaking change, but I hope you find the upgrade not too difficult as the improvements are significant.

### Major Changes

* Complete internal re-architecture for massive performance improvements. See below for details.
* **Integer axes as well as float:** There are now two high-level versions of the library: `float`, which (like the previous versions) uses float types for the positions of points, and `fixed`, which uses either integer or fixed point representation for the position of points.
* **integer-only contents**: The contents of the tree is restricted to being integer values only now - usually usize or u32, but you could go down to u16 or even u8 if you want. This simplifies some aspects of the internals. It has always been much more performant to store content objects in a separate Vec or array, and only store an index into this array in Kiddo itself; this change now makes that the only way.
* **Generic bucket size:** The bucket size is now a generic parameter. This allows us to use `Array`s for the bucket contents rather than `Vec`s, preventing the need for a second indirection and allowing all the memory for an entire tree to be allocated up-front in a single allocation at creation time if the required capacity of the tree is known beforehand.
* **Generic bucket index:** There is an underlying struct that additionally has a 5th generic parameter to determine the integer type that is used to index the buckets internally. Choosing a type for this param that is just small enough to permit the maximum number of buckets that your use case will encounter can give a performance boost that is especially noticeable when storing 100k -> 10M items by permitting many more of the index nodes to fit inside the CPU cache.
* **`rkyv` feature:** Previous versions provided the `serde` feature for serialization and deserialization. Due to the large number of memory allocations that were needed for big trees though, this could be quite slow - my primary use case for which Kiddo was created makes use of a ~1Gb 15-million-node tree, which took anywhere between 9 seconds to deserialize for a quick desktop all the way up to around 30s for an AWS Lambda function. Whilst in v2 the `serde` functionality is still there, you can now try an alternative approach using the incredibly quick `rkyv` zero-copy deserialization library. Amazingly, this new feature reduced the deserialization time for the use-case above to 0.6s - and the vast majority of that time is just the overhead of memory-mapping the raw file from SSD into memory.
* **`select_nth_unstable_by` node splits**: kd trees need to split the contents of a bucket between two new buckets once a bucket gets full. Previously, Kiddo would fully sort the contents of the bucket to be split, but we don't need the list to be *fully* sorted - we only care that the "smaller" half of the items come before the "larger" ones, not the order within those two groups. Rust has a `select_nth_unstable_by` function that can do this, enabling nodes to be split more quickly. This is complicated by the architectural change mentioned below that changes Leaf nodes to use separate Arrays to store points compared to contents. We need to run `select_nth_unstable_by` over the points array but then apply the same actions that were taken to sort that array to the contents array. This required the development of a custom version of `select_nth_unstable_by` that applies the same sort actions made on a "main" array to a "mirror" array.
* **unchecked indexing of nodes**: access into the node Vecs is now unchecked, to gain a performance improvement by eliminating bounds checking.
* **empty / full and NaN checks removed**: Pre-v2 Kiddo had a lot of checks to ensure that it behaved well when items with NaNs inside their positions were attempted to be inserted, or when queries were made against empty trees. If you need these kind of checks, **This is now your responsibility** - check for NaNs and empty trees prior to insertion / query. For anyone that does not need these checks, you will benefit from better performance by not having them performed.


### Architectural changes

* Previously, the node tree structure was pointer-based, i.e. node structs had `Box<>`ed references to `left` and `right` child nodes. Kiddo v2 moves to an index-based approach: a top-level KdTree struct stores nodes in a Vec<>, and each node's `left` and `right` properties are indexes into this `Vec<>` instead. This gives a few advantages:
  * Creating a `KdTree` with a known up-front capacity can pre-allocate all nodes in a single dynamic allocation, rather than requiring one allocation per node. Even in the case where the number of nodes is not known in advance, `Vec<>` grows in chunks rather than an item at a time, resulting again in far fewer dynamic allocations being required.
* Prior to v2, Kiddo used a single Node struct that was implemented as an enum, with one enum version being used for Stem node and one for Leaf nodes. v2 switches to having separate Leaf node and Stem node structs, each being stored in their own Vec<> inside the KdTree struct. This gives several advantages:
  * Stem nodes can be made as small as possible by choosing generic types that minimize the size of each Stem node. This ensures that more of them can fit into the CPU cache.
  * Stem nodes are all next to each other and accessed in ascending order, initially close to sequentially, again helping to improve cache utilisation during the stem traversal phase of a query.
  * Leaf nodes can now use arrays to store their contents rather than Vec<>s, cutting down on multiple indirection. Due to the previous implementation using an Enum for both node types, this would have wasted too much space.
* Leaf nodes used to store a vec of structs, with each struct containing the coordinates and content for each item stored in the tree. With v2, leaf nodes now keeps two separate arrays to store their items - one for the points themselves, and one for the content. This means that the points are contiguous in memory which makes it easier for the compiler to autovectorise iteration across the points during queries, improving query performance.
* Leaf and Stem nodes now no longer store max and min bounds of their contents. The same technique as nabo-rs is used instead, whereby the distance from the most distant current result point to the most recent 3 split planes is used. This massively reduces the size of the stem nodes, so that much more of them fit in the CPU cache. Construction time is also slightly quicker thanks to not needing to calculate bounds.
