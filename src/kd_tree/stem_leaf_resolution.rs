#![cfg_attr(feature = "rkyv_08", allow(missing_docs))]

use nonmax::NonMaxUsize;

use super::{
    resolve_arithmetic_terminal_stem_idx, resolve_mapped_terminal_stem_idx, StemLeafResolution,
};

/// Owned strategy for resolving stem indices to leaf indices during traversal.
///
/// Different variants optimize for different tree usage patterns:
/// - `Arithmetic`: For immutable trees where all leaves are at the same depth
/// - `Pristine`: For mutable trees that haven't had structural mutations yet
/// - `Mapped`: For mutable trees after leaf splits/merges have occurred
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
#[cfg_attr(feature = "rkyv_08", rkyv(attr(allow(missing_docs))))]
#[cfg_attr(
    feature = "rkyv_08",
    rkyv(archived = ArchivedStemLeafResolution)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedStemLeafResolution {
    /// Immutable strategies: leaf index can be calculated arithmetically.
    ///
    /// All leaves are guaranteed to be at the same depth, so leaf indices
    /// can be computed directly from stem indices.
    Arithmetic {
        /// how many levels deep the stem tree is
        stems_depth: usize,
        /// how many leaves there are
        leaf_count: usize,
    },
    /// Mutable strategies in pristine state: no structural mutations yet.
    ///
    /// Uses arithmetic resolution like `Arithmetic`, but can transition
    /// to `Mapped` when the first leaf split/merge occurs.
    Pristine {
        /// initial stem depth
        stems_depth: usize,
        /// how many leaves there are initially
        leaf_count: usize,
    },
    /// Mutable strategies after structural mutations (split/merge).
    ///
    /// Requires explicit mapping from terminal stem indices to leaf indices
    /// because leaves may be at different depths.
    Mapped {
        /// Index of the first stem that might point to a leaf
        min_stem_leaf_idx: usize,
        /// Maps stem indices to leaf indices.
        /// `None` means the stem has children, `Some(idx)` means it points to leaf `idx`.
        #[cfg_attr(
            feature = "serde",
            serde(with = "crate::custom_serde::option_nonmax_usize_vec")
        )]
        #[cfg_attr(
            feature = "rkyv_08",
            rkyv(with = rkyv_08::with::Map<crate::rkyv::adapters::OptionNonMaxUsizeAsUsize>)
        )]
        leaf_idx_map: Vec<Option<NonMaxUsize>>,
    },
}

impl StemLeafResolution for OwnedStemLeafResolution {
    #[inline(always)]
    fn uses_arithmetic(&self) -> bool {
        matches!(self, Self::Arithmetic { .. } | Self::Pristine { .. })
    }

    #[inline(always)]
    fn resolve_terminal_stem_idx(&self, stem_idx: usize, arithmetic_leaf_idx: usize) -> usize {
        match self {
            Self::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => resolve_mapped_terminal_stem_idx(
                stem_idx,
                *min_stem_leaf_idx,
                leaf_idx_map.len(),
                |map_idx| {
                    leaf_idx_map
                        .get(map_idx)
                        .and_then(|opt| opt.map(|n| n.get()))
                },
            ),
            Self::Arithmetic {
                stems_depth,
                leaf_count,
            }
            | Self::Pristine {
                stems_depth,
                leaf_count,
            } => resolve_arithmetic_terminal_stem_idx(
                stem_idx,
                arithmetic_leaf_idx,
                *stems_depth,
                *leaf_count,
            ),
        }
    }

    #[inline(always)]
    fn is_terminal_stem_idx(&self, stem_idx: usize) -> bool {
        match self {
            Self::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } if stem_idx >= *min_stem_leaf_idx => {
                let map_idx = stem_idx - *min_stem_leaf_idx;
                leaf_idx_map.get(map_idx).is_some_and(Option::is_some)
            }
            _ => false,
        }
    }
}

impl OwnedStemLeafResolution {
    /// Returns true if this resolution strategy uses arithmetic leaf-index resolution.
    #[inline(always)]
    pub fn uses_arithmetic(&self) -> bool {
        <Self as StemLeafResolution>::uses_arithmetic(self)
    }

    /// Resolves a terminal stem index to a leaf index.
    #[inline(always)]
    pub fn resolve_terminal_stem_idx(&self, stem_idx: usize, arithmetic_leaf_idx: usize) -> usize {
        <Self as StemLeafResolution>::resolve_terminal_stem_idx(self, stem_idx, arithmetic_leaf_idx)
    }

    /// Returns true if `stem_idx` maps directly to a leaf.
    #[inline(always)]
    pub fn is_terminal_stem_idx(&self, stem_idx: usize) -> bool {
        <Self as StemLeafResolution>::is_terminal_stem_idx(self, stem_idx)
    }
}

#[cfg(feature = "rkyv_08")]
impl StemLeafResolution for ArchivedStemLeafResolution {
    #[inline(always)]
    fn uses_arithmetic(&self) -> bool {
        matches!(self, Self::Arithmetic { .. } | Self::Pristine { .. })
    }

    #[inline(always)]
    fn resolve_terminal_stem_idx(&self, stem_idx: usize, arithmetic_leaf_idx: usize) -> usize {
        match self {
            Self::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => resolve_mapped_terminal_stem_idx(
                stem_idx,
                min_stem_leaf_idx.to_native() as usize,
                leaf_idx_map.len(),
                |map_idx| {
                    let value = leaf_idx_map.get(map_idx)?.to_native() as usize;
                    crate::rkyv::adapters::archived_option_nonmax_usize_is_some(value)
                        .then_some(value)
                },
            ),
            Self::Arithmetic {
                stems_depth,
                leaf_count,
            }
            | Self::Pristine {
                stems_depth,
                leaf_count,
            } => resolve_arithmetic_terminal_stem_idx(
                stem_idx,
                arithmetic_leaf_idx,
                stems_depth.to_native() as usize,
                leaf_count.to_native() as usize,
            ),
        }
    }

    #[inline(always)]
    fn is_terminal_stem_idx(&self, stem_idx: usize) -> bool {
        match self {
            Self::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => {
                let min_stem_leaf_idx = min_stem_leaf_idx.to_native() as usize;
                if stem_idx >= min_stem_leaf_idx {
                    let map_idx = stem_idx - min_stem_leaf_idx;
                    leaf_idx_map.get(map_idx).is_some_and(|value| {
                        crate::rkyv::adapters::archived_option_nonmax_usize_is_some(
                            value.to_native() as usize,
                        )
                    })
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}
