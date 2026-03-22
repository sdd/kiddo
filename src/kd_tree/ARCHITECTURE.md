# v6 KdTree Architecture

This document describes the architecture of the v6 "unified tree" implementation in `src/kd_tree/`,
which provides a flexible, trait-based k-d tree that supports both mutable and immutable patterns
with pluggable strategies for stem ordering and leaf storage.

## Design Goals

1. **Unify mutable and immutable patterns** - Single codebase supporting both use cases
2. **Pluggable strategies** - Swappable stem ordering (Eytzinger, Donnelly variants) and leaf storage strategies
3. **Zero-cost abstractions** - Static dispatch via generics, no `dyn Trait` in hot paths
4. **Type safety** - Compile-time enforcement of valid configurations
5. **Maintainability** - Clear separation of concerns between tree structure, query logic, and storage

## Core Components

### Tree Structure (`src/kd_tree/mod.rs`)

The main `KdTree` type is parameterized by strategies:

```rust
pub struct KdTree<A, T, SS, LS, const K: usize>
where
    SS: StemStrategy,
    LS: LeafStrategy<A, T, K>,
{
    stem_strategy: SS,
    leaf_strategy: LS,
    size: usize,
    // ...
}
```

**Key parameters:**
- `A`: Axis/coordinate type (f32, f64, fixed-point, etc.)
- `T`: Data payload type
- `SS`: Stem strategy (determines internal node layout)
- `LS`: Leaf strategy (determines point storage)
- `K`: Dimensionality (compile-time constant)

### Stem Strategies (`src/stem_strategies/`)

Stem strategies control how internal (non-leaf) nodes are laid out in memory. This is critical for cache performance.

**The `StemStrategy` trait** (`src/traits.rs`):

```rust
pub trait StemStrategy {
    type Stem;  // The stored stem node type
    type Stack<O>: StackTrait<O, Self>;  // Stack for query backtracking

    // Navigation
    fn get_child_stem_idx(&self, parent_idx: usize, ...) -> StemLeafResolution;
    fn get_leaf_idx(&self, stem_idx: usize, ...) -> usize;

    // Query methods
    fn nearest_one<...>(&self, ...) -> (Distance, Payload);
    fn backtracking_query_with_stack<...>(&self, ...) -> QueryResult;
    // ... other query methods
}
```

**Key concept: `StemLeafResolution`**

Stem navigation returns a `StemLeafResolution` enum:

```rust
pub enum StemLeafResolution {
    Stem(usize),      // Continue traversal to stem node at index
    Leaf(usize),      // Reached a leaf at index
    LeafByMapping,    // Leaf index computed arithmetically (Eytzinger-style)
}
```

This enum handles two different tree termination patterns:

1. **Arithmetic mapping** (Eytzinger): Leaf index can be computed directly from stem index using arithmetic (e.g., `stem_idx - num_stems`)
2. **Explicit mapping** (Donnelly): Leaf index is stored explicitly in the stem node

**Available strategies:**

| Strategy | Layout | Cache Optimization | Leaf Termination |
|----------|--------|-------------------|------------------|
| `Eytzinger` | Breadth-first (BFS) | Good sequential access | Arithmetic (`LeafByMapping`) |
| `Donnelly2` | Cache-aware blocks | Optimized for cache lines | Explicit (`Leaf(idx)`) |
| `Donnelly2PF` | Donnelly + prefetch | Hardware prefetch hints | Explicit |
| `Donnelly2BlockMarkerSimd` | Donnelly + SIMD | Block-at-once traversal | Explicit |
| `Donnelly3` | Software prefetch | Manual prefetch | Explicit |

### Leaf Strategies (`src/kd_tree/leaf_strategies/`)

Leaf strategies control how points and payloads are stored at the tree's leaves.

**The `LeafStrategy` trait** (`src/kd_tree/leaf_strategies/mod.rs`):

```rust
pub trait LeafStrategy<A, T, const K: usize> {
    // Point access
    fn get_point(&self, leaf_idx: usize) -> [A; K];
    fn get_point_ref(&self, leaf_idx: usize) -> &[A; K];

    // Payload access
    fn get_item(&self, leaf_idx: usize) -> T;
    fn get_item_ref(&self, leaf_idx: usize) -> &T;

    // Construction
    fn from_leaves(leaves: Vec<([A; K], T)>) -> Self;

    // Size info
    fn len(&self) -> usize;
}
```

**Available strategies:**

1. **`VecOfArrays`** (`vec_of_arrays.rs`):
   - Stores points column-wise: `points: [Vec<A>; K]`
   - Better cache locality when accessing many points in same dimension
   - Preferred for immutable trees and read-heavy workloads
   - Used by `ImmutableKdTree`

2. **`LeafNodes`** (`leaf_nodes.rs`):
   - Traditional row-wise storage: `Vec<LeafNode<A, T, K>>`
   - Each `LeafNode` contains `point: [A; K]` and `item: T`
   - Simpler memory layout, better for mutation
   - Used by mutable `KdTree`

### Query System (`src/kd_tree/query/`)

Query implementations are modularized by query type:

- `nearest_one.rs` - Single nearest neighbor
- `nearest_n.rs` - K-nearest neighbors
- `within.rs` - All points within radius
- `within_unsorted.rs` - Points within radius (unsorted)
- `best_n_within.rs` - Best N points within radius
- `nearest_n_within.rs` - K-nearest within radius

**Query Orchestrator** (`src/kd_tree/query_orchestrator.rs`):

The orchestrator coordinates query execution across different stem strategies. It provides:

1. **Stack-based backtracking**: Maintains a stack of nodes to revisit
2. **Pruning**: Eliminates branches that can't contain better results
3. **Distance tracking**: Maintains "rd" (remaining distance) arrays for efficient pruning

### Distance Metrics (`src/distance/`)

Distance metrics are implemented via traits from `src/traits_unified_2.rs`:

```rust
pub trait DistanceMetricUnified<A, const K: usize> {
    type Output;  // Distance result type (may differ from A)

    fn dist(&self, a: &[A; K], b: &[A; K]) -> Self::Output;
    fn dist1(&self, a: A, b: A) -> Self::Output;  // 1D distance
}
```

**Available metrics:**
- `SquaredEuclidean` - Most common, avoids sqrt
- `Manhattan` - L1 distance
- `DotProduct` - For similarity search (not a true metric)

**Distance widening**: The `Output` associated type allows distance calculations to use higher precision than coordinates:
- `f32` coordinates → `f64` distances
- `FixedI32<U16>` coordinates → `FixedI64<U32>` distances

This prevents overflow and maintains precision in distance calculations.

## SIMD Optimization (`src/kd_tree/query_orchestrator/simd/`)

For Donnelly block-based strategies, SIMD optimizations accelerate pruning:

**`SimdPrune` trait** (`src/stem_strategies/donnelly_2_blockmarker_simd/prune_traits.rs`):

```rust
pub trait SimdPrune: AxisUnified<Coord = Self> {
    fn simd_prune_block3(
        rd_values: &[Self; 8],
        max_dist: Self,
        sibling_mask: u8,
    ) -> u8;
}
```

**Type-specific dispatch**: Instead of `size_of`-based dispatch (which can't distinguish f32 from FixedI32<U0>), pruning is implemented per concrete type:

- **f32/f64**: AVX2 (`_mm256_cmp_ps/pd`) or NEON (`vcleq_f32/f64`) intrinsics
- **Fixed-point**: Autovec fallback currently; future integer SIMD
- **f16**: Autovec fallback currently; future widening to f32 SIMD

**Autovec fallback**: When SIMD intrinsics aren't available, a simple loop allows the compiler to auto-vectorize:

```rust
let mut mask: u8 = 0;
for i in 0..8 {
    if rd_values[i] <= max_dist {
        mask |= 1 << i;
    }
}
mask & sibling_mask
```

**Block sizes**: The trait is currently `simd_prune_block3` (8 children, u8 mask). Future work will generalize to Block4 (16 children, u16 mask) and Block5 (32 children, u32 mask).

## Data Flow: Query Execution

### Nearest-One Query Example

1. **Entry point** (`src/kd_tree/query/nearest_one.rs`):
   ```rust
   pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
   ```

2. **Delegation to stem strategy** (`src/traits.rs`):
   ```rust
   self.stem_strategy.nearest_one::<A, T, D::Output, D, _, LS, K, B>(
       query,
       &self.leaf_strategy,
       &self.stem_to_leaf_map,
   )
   ```

3. **Strategy-specific traversal** (e.g., Donnelly block-based):
   - Traverse internal nodes using `get_child_stem_idx()`
   - Build backtracking stack during descent
   - Check for `StemLeafResolution::Leaf` or `LeafByMapping`

4. **Leaf lookup**:
   - Use `LeafStrategy::get_point()` to retrieve point
   - Use `LeafStrategy::get_item()` to retrieve payload
   - Compute exact distance with `DistanceMetric::dist()`

5. **Backtracking** (`src/kd_tree/query_orchestrator.rs`):
   - Pop nodes from stack
   - Prune using SIMD if available: `O::simd_prune_block3()`
   - Continue exploring unpruned children
   - Update best result when better distance found

6. **Return result**: `(best_distance, payload)`

## Tree Construction

### Immutable Tree (`ImmutableKdTree`)

1. Points collected into vec
2. Recursive partitioning to determine stem structure and split values
3. Stems laid out according to strategy (Eytzinger/Donnelly)
4. Leaves stored column-wise via `VecOfArrays` strategy
5. Optional alignment for SIMD operations
6. Optional serialization with `rkyv` or `serde`

### Mutable Tree (`KdTree` in `src/mutable/`)

1. Points inserted one at a time or in bulk
2. Stems stored as index-linked binary tree
3. Leaves stored row-wise via `LeafNodes` strategy
4. Rebalancing not currently implemented (degrades to list with sequential inserts)

## Type-Level Safety

### Sealed Traits

Many traits are sealed to prevent external implementations:

```rust
mod sealed {
    pub trait Sealed {}
}

pub trait SimdPrune: sealed::Sealed {
    // ...
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}
```

This ensures only tested, supported types can implement critical traits.

### Static Dispatch

All hot-path code uses static dispatch via generics:

```rust
pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
where
    D: DistanceMetricUnified<A, K>,
    D::Output: SimdPrune,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
{
    // Monomorphized for each (A, T, D, SS, LS) combination
}
```

No `dyn Trait` objects, no runtime dispatch overhead.

### Feature Gates

Optional functionality is gated by Cargo features:

- `#[cfg(feature = "simd")]` - SIMD intrinsics (nightly-only)
- `#[cfg(feature = "fixed")]` - Fixed-point support
- `#[cfg(feature = "f16")]` - Half-precision float support
- `#[cfg(feature = "rkyv_08")]` - Zero-copy serialization

## Memory Layout Considerations

### Cache-Line Alignment

Donnelly strategies align blocks to cache line boundaries (64 bytes):

```rust
#[repr(C, align(64))]
pub struct DonnellyBlock3<A> {
    pivots: [A; 7],
    // ... other fields
}
```

This ensures each block traversal is a single cache-line fetch.

### Column vs Row Storage

**Column storage** (`VecOfArrays`):
```
X: [1.0, 2.0, 3.0, 4.0, ...]
Y: [5.0, 6.0, 7.0, 8.0, ...]
Z: [9.0, 10.0, 11.0, 12.0, ...]
```

Benefits:
- Better cache locality when processing many points in same dimension
- SIMD operations can load contiguous dimension values

**Row storage** (`LeafNodes`):
```
[(1.0, 5.0, 9.0), (2.0, 6.0, 10.0), (3.0, 7.0, 11.0), ...]
```

Benefits:
- Simpler mental model
- Better for single-point operations

## Extension Points

### Adding a New Stem Strategy

1. Define stem node type with necessary split/navigation data
2. Implement `StemStrategy` trait
3. Key methods:
   - `get_child_stem_idx()` - Navigate from parent to child
   - `get_leaf_idx()` - Resolve leaf when traversal terminates
   - Choose appropriate `StemLeafResolution` variant
4. Implement `backtracking_query_with_stack()` for query support
5. Add tests comparing against Eytzinger baseline

### Adding a New Leaf Strategy

1. Define storage type (Vec, HashMap, etc.)
2. Implement `LeafStrategy` trait
3. Key methods:
   - `get_point()` / `get_point_ref()` - Retrieve coordinates
   - `get_item()` / `get_item_ref()` - Retrieve payload
   - `from_leaves()` - Construct from sorted leaf data
4. Add tests for construction and access

### Adding a New Distance Metric

1. Implement `DistanceMetricUnified<A, K>` trait
2. Define `Output` type (may be wider than `A`)
3. Implement `dist()` for K-dimensional distance
4. Implement `dist1()` for 1-dimensional distance (used in pruning)
5. For Donnelly strategies, ensure `Output` implements `SimdPrune`
6. Add tests across different axis types

## Current Limitations and Future Work

### Current Limitations

1. **Mutable tree rebalancing**: Not implemented; performance degrades with sequential insertions
2. **Block4/Block5 generalization**: Only Block3 fully implemented for Donnelly SIMD
3. **Integer SIMD**: Fixed-point types use autovec fallback, not hand-tuned intrinsics
4. **f16 SIMD**: Currently autovec; could widen to f32 for better performance
5. **DotProduct metric**: Not properly supported in pruning logic (needs similarity bounds, not distance bounds)

### Planned Improvements (See DEVELOPMENT_PLAN.md)

1. **Remove remaining `size_of` dispatch**: In block comparison logic (CompareBlock3/4)
2. **Interval-based Block4/Block5**: Correct pruning semantics for larger blocks
3. **Metric generalization**: Proper support for Manhattan and DotProduct
4. **Mask width generalization**: Support u16/u32 masks for Block4/Block5
5. **SIMD extensions**: Integer SIMD for fixed-point, NEON parity with x86

## References

**Key files:**
- `src/kd_tree/mod.rs` - Main tree structure
- `src/traits.rs` - Core trait definitions (`StemStrategy`, etc.)
- `src/kd_tree/leaf_strategies/mod.rs` - Leaf storage abstraction
- `src/stem_strategies/` - Stem ordering implementations
- `src/kd_tree/query_orchestrator.rs` - Query coordination
- `src/kd_tree/query_orchestrator/simd/mod.rs` - SIMD pruning
- `src/stem_strategies/donnelly_2_blockmarker_simd/prune_traits.rs` - Type-specific SIMD dispatch

**Related documentation:**
- `DEVELOPMENT_PLAN.md` - Ongoing generalization work
- `AGENTS.md` - Project overview and build instructions
- Root `README.md` - User-facing documentation
