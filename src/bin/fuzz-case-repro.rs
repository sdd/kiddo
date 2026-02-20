use std::array;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::env;
use std::num::NonZeroUsize;

use kiddo::distance::float::{Manhattan, SquaredEuclidean};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
use kiddo::kd_tree::KdTree as V6KdTree;
use kiddo::mutable::float::kdtree::KdTree;
use kiddo::nearest_neighbour::NearestNeighbour;
use kiddo::stem_strategies::{Donnelly, Eytzinger};
use kiddo::traits::{Axis, DistanceMetric};
use kiddo::traits_unified_2::{Manhattan as V6Manhattan, SquaredEuclidean as V6SquaredEuclidean};
use kiddo::StemStrategy;

#[cfg(feature = "simd")]
use kiddo::stem_strategies::{Block3, Block4, DonnellyMarkerSimd};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_MIN_POW: u32 = 10;
const DEFAULT_MAX_POW: u32 = 24;
const DEFAULT_PERTURB_MIN: i32 = -5;
const DEFAULT_PERTURB_MAX: i32 = 5;
const DEFAULT_MAX_NEAREST_N: usize = 32;

#[derive(Clone, Copy)]
struct FuzzConfig {
    min_pow: u32,
    max_pow: u32,
    perturb_min: i32,
    perturb_max: i32,
    max_nearest_n: usize,
}

#[derive(Debug)]
struct ReproParams {
    kind: String,
    leaf: Option<String>,
    scalar: String,
    strategy: String,
    b: usize,
    k: usize,
    size: usize,
    content_seed: u64,
    query_seed: u64,
}

fn main() {
    let arg = env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "Usage: cargo run --bin fuzz-case-repro -- <repro-id>\n\
             Example: failure-kind_immutable-ty_f32-strategy_donnelly-b_32-k_4-size_12345-content_seed_1-query_seed_2\n\
             Example (v6): failure-kind_v6_mutable-leaf_vec_of_arrays-ty_f32-strategy_eytzinger-b_32-k_4-size_12345-content_seed_1-query_seed_2"
        );
        std::process::exit(2);
    });

    let params = match parse_repro_id(&arg) {
        Ok(params) => params,
        Err(err) => {
            eprintln!("Invalid repro id: {err}");
            std::process::exit(2);
        }
    };

    if let Err(err) = run_repro(&params) {
        eprintln!("Repro failed: {err}");
        std::process::exit(1);
    }
}

fn parse_repro_id(input: &str) -> Result<ReproParams, String> {
    let trimmed = input.trim();
    let trimmed = trimmed.strip_prefix("repro=").unwrap_or(trimmed);

    if !trimmed.starts_with("failure-") {
        return Err("repro id must start with 'failure-'".to_string());
    }

    let parts: Vec<&str> = trimmed.split('-').collect();
    let mut fields: HashMap<&str, &str> = HashMap::new();
    let keys = [
        "content_seed",
        "query_seed",
        "strategy",
        "kind",
        "leaf",
        "ty",
        "size",
        "b",
        "k",
    ];

    for part in parts.iter().skip(1) {
        let mut matched = None;
        for key in keys {
            if let Some(rest) = part.strip_prefix(key) {
                if let Some(value) = rest.strip_prefix('_') {
                    matched = Some((key, value));
                    break;
                }
            }
        }

        let (key, value) =
            matched.ok_or_else(|| format!("missing key/value in segment '{part}'"))?;
        fields.insert(key, value);
    }

    let kind = normalize_kind(required_field(&fields, "kind")?);
    let scalar = required_field(&fields, "ty")?.to_ascii_lowercase();
    let strategy = normalize_strategy(required_field(&fields, "strategy")?);
    let leaf = fields.get("leaf").map(|value| value.to_ascii_lowercase());

    let b = parse_usize(required_field(&fields, "b")?, "b")?;
    let k = parse_usize(required_field(&fields, "k")?, "k")?;
    let size = parse_usize(required_field(&fields, "size")?, "size")?;
    let content_seed = parse_u64(required_field(&fields, "content_seed")?, "content_seed")?;
    let query_seed = parse_u64(required_field(&fields, "query_seed")?, "query_seed")?;

    if kind.starts_with("v6_") && leaf.is_none() {
        return Err("v6 repro ids must include leaf_* field".to_string());
    }

    Ok(ReproParams {
        kind,
        leaf,
        scalar,
        strategy,
        b,
        k,
        size,
        content_seed,
        query_seed,
    })
}

fn normalize_kind(value: &str) -> String {
    value.to_ascii_lowercase().replace('-', "_")
}

fn normalize_strategy(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    match lower.as_str() {
        "donnellysimd" | "donnelly_simd" | "donnelly-simd" => "donnelly_simd".to_string(),
        "donnellysimdblock3" | "donnelly_simd_block3" | "donnelly-simd-block3" => {
            "donnelly_simd_block3".to_string()
        }
        "donnellysimdblock4" | "donnelly_simd_block4" | "donnelly-simd-block4" => {
            "donnelly_simd_block4".to_string()
        }
        other => other.to_string(),
    }
}

fn required_field<'a>(fields: &HashMap<&'a str, &'a str>, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| format!("missing field '{key}'"))
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid {label} value '{value}'"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid {label} value '{value}'"))
}

fn run_repro(params: &ReproParams) -> Result<(), String> {
    match (params.kind.as_str(), params.scalar.as_str()) {
        ("mutable", "f32") => run_mutable_f32(params),
        ("mutable", "f64") => run_mutable_f64(params),
        ("immutable", "f32") => run_immutable_f32(params),
        ("immutable", "f64") => run_immutable_f64(params),
        ("v6_mutable", "f32") => run_v6_mutable_f32(params),
        ("v6_mutable", "f64") => run_v6_mutable_f64(params),
        ("v6_immutable", "f32") => run_v6_immutable_f32(params),
        ("v6_immutable", "f64") => run_v6_immutable_f64(params),
        _ => Err(format!(
            "unsupported kind/scalar: {} {}",
            params.kind, params.scalar
        )),
    }
}

fn run_mutable_f32(params: &ReproParams) -> Result<(), String> {
    match (params.k, params.b) {
        (2, 16) => run_mutable_case_f32::<2, 16>(params),
        (2, 32) => run_mutable_case_f32::<2, 32>(params),
        (2, 64) => run_mutable_case_f32::<2, 64>(params),
        (3, 16) => run_mutable_case_f32::<3, 16>(params),
        (3, 32) => run_mutable_case_f32::<3, 32>(params),
        (3, 64) => run_mutable_case_f32::<3, 64>(params),
        (4, 16) => run_mutable_case_f32::<4, 16>(params),
        (4, 32) => run_mutable_case_f32::<4, 32>(params),
        (4, 64) => run_mutable_case_f32::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for mutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_mutable_f64(params: &ReproParams) -> Result<(), String> {
    match (params.k, params.b) {
        (2, 16) => run_mutable_case_f64::<2, 16>(params),
        (2, 32) => run_mutable_case_f64::<2, 32>(params),
        (2, 64) => run_mutable_case_f64::<2, 64>(params),
        (3, 16) => run_mutable_case_f64::<3, 16>(params),
        (3, 32) => run_mutable_case_f64::<3, 32>(params),
        (3, 64) => run_mutable_case_f64::<3, 64>(params),
        (4, 16) => run_mutable_case_f64::<4, 16>(params),
        (4, 32) => run_mutable_case_f64::<4, 32>(params),
        (4, 64) => run_mutable_case_f64::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for mutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_immutable_f32(params: &ReproParams) -> Result<(), String> {
    match params.strategy.as_str() {
        "eytzinger" => run_immutable_strategy_f32::<EytzingerStrategyF32>(params),
        "donnelly" => run_immutable_strategy_f32::<DonnellyStrategyF32>(params),
        "donnelly_simd" => run_immutable_strategy_f32::<DonnellySimdStrategyF32>(params),
        _ => Err(format!(
            "unsupported immutable strategy '{}'",
            params.strategy
        )),
    }
}

fn run_immutable_f64(params: &ReproParams) -> Result<(), String> {
    match params.strategy.as_str() {
        "eytzinger" => run_immutable_strategy_f64::<EytzingerStrategyF64>(params),
        "donnelly" => run_immutable_strategy_f64::<DonnellyStrategyF64>(params),
        "donnelly_simd" => run_immutable_strategy_f64::<DonnellySimdStrategyF64>(params),
        _ => Err(format!(
            "unsupported immutable strategy '{}'",
            params.strategy
        )),
    }
}

fn run_v6_mutable_f32(params: &ReproParams) -> Result<(), String> {
    ensure_leaf(params, "vec_of_arrays")?;
    match params.strategy.as_str() {
        "eytzinger" => run_v6_mutable_strategy_f32::<V6EytzingerStrategyF32>(params),
        "donnelly" => run_v6_mutable_strategy_f32::<V6DonnellyStrategyF32>(params),
        "donnelly_simd" => run_v6_mutable_strategy_f32::<V6DonnellySimdStrategyF32>(params),
        "donnelly_simd_block3" => Err(
            "donnelly_simd_block3 is not supported for f32 on 64-byte-line targets; use donnelly_simd_block4"
                .to_string(),
        ),
        "donnelly_simd_block4" => {
            run_v6_mutable_strategy_f32::<V6DonnellySimdBlock4StrategyF32>(params)
        }
        _ => Err(format!(
            "unsupported v6 mutable strategy '{}'",
            params.strategy
        )),
    }
}

fn run_v6_mutable_f64(params: &ReproParams) -> Result<(), String> {
    ensure_leaf(params, "vec_of_arrays")?;
    match params.strategy.as_str() {
        "eytzinger" => run_v6_mutable_strategy_f64::<V6EytzingerStrategyF64>(params),
        "donnelly" => run_v6_mutable_strategy_f64::<V6DonnellyStrategyF64>(params),
        "donnelly_simd" => run_v6_mutable_strategy_f64::<V6DonnellySimdStrategyF64>(params),
        "donnelly_simd_block3" => {
            run_v6_mutable_strategy_f64::<V6DonnellySimdBlock3StrategyF64>(params)
        }
        "donnelly_simd_block4" => Err(
            "donnelly_simd_block4 is not supported for f64 on 64-byte-line targets; use donnelly_simd_block3"
                .to_string(),
        ),
        _ => Err(format!(
            "unsupported v6 mutable strategy '{}'",
            params.strategy
        )),
    }
}

fn run_v6_immutable_f32(params: &ReproParams) -> Result<(), String> {
    ensure_leaf(params, "flat_vec")?;
    match params.strategy.as_str() {
        "eytzinger" => run_v6_immutable_strategy_f32::<V6EytzingerStrategyF32>(params),
        "donnelly" => run_v6_immutable_strategy_f32::<V6DonnellyStrategyF32>(params),
        "donnelly_simd" => run_v6_immutable_strategy_f32::<V6DonnellySimdStrategyF32>(params),
        "donnelly_simd_block3" => Err(
            "donnelly_simd_block3 is not supported for f32 on 64-byte-line targets; use donnelly_simd_block4"
                .to_string(),
        ),
        "donnelly_simd_block4" => {
            run_v6_immutable_strategy_f32::<V6DonnellySimdBlock4StrategyF32>(params)
        }
        _ => Err(format!(
            "unsupported v6 immutable strategy '{}'",
            params.strategy
        )),
    }
}

fn run_v6_immutable_f64(params: &ReproParams) -> Result<(), String> {
    ensure_leaf(params, "flat_vec")?;
    match params.strategy.as_str() {
        "eytzinger" => run_v6_immutable_strategy_f64::<V6EytzingerStrategyF64>(params),
        "donnelly" => run_v6_immutable_strategy_f64::<V6DonnellyStrategyF64>(params),
        "donnelly_simd" => run_v6_immutable_strategy_f64::<V6DonnellySimdStrategyF64>(params),
        "donnelly_simd_block3" => {
            run_v6_immutable_strategy_f64::<V6DonnellySimdBlock3StrategyF64>(params)
        }
        "donnelly_simd_block4" => Err(
            "donnelly_simd_block4 is not supported for f64 on 64-byte-line targets; use donnelly_simd_block3"
                .to_string(),
        ),
        _ => Err(format!(
            "unsupported v6 immutable strategy '{}'",
            params.strategy
        )),
    }
}

trait StrategySelectorF32 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
}

trait StrategySelectorF64 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
}

struct EytzingerStrategyF32;
struct DonnellyStrategyF32;
struct DonnellySimdStrategyF32;
struct EytzingerStrategyF64;
struct DonnellyStrategyF64;
struct DonnellySimdStrategyF64;

impl StrategySelectorF32 for EytzingerStrategyF32 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_immutable_case_f32::<K, B, Eytzinger<K>>(params)
    }
}

impl StrategySelectorF32 for DonnellyStrategyF32 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_immutable_case_f32::<K, B, Donnelly<4, 64, 4, K>>(params)
    }
}

impl StrategySelectorF32 for DonnellySimdStrategyF32 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_immutable_case_f32::<K, B, DonnellyMarkerSimd<Block4, 64, 4, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }
}

impl StrategySelectorF64 for EytzingerStrategyF64 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_immutable_case_f64::<K, B, Eytzinger<K>>(params)
    }
}

impl StrategySelectorF64 for DonnellyStrategyF64 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_immutable_case_f64::<K, B, Donnelly<3, 64, 8, K>>(params)
    }
}

impl StrategySelectorF64 for DonnellySimdStrategyF64 {
    fn run<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_immutable_case_f64::<K, B, DonnellyMarkerSimd<Block3, 64, 8, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }
}

fn ensure_leaf(params: &ReproParams, expected: &str) -> Result<(), String> {
    match params.leaf.as_deref() {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(format!(
            "unsupported leaf strategy '{value}' for kind {} (expected {expected})",
            params.kind
        )),
        None => Err(format!(
            "missing leaf strategy for kind {} (expected {expected})",
            params.kind
        )),
    }
}

trait V6StrategySelectorF32 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
}

trait V6StrategySelectorF64 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String>;
}

struct V6EytzingerStrategyF32;
struct V6DonnellyStrategyF32;
struct V6DonnellySimdStrategyF32;
struct V6DonnellySimdBlock4StrategyF32;
struct V6EytzingerStrategyF64;
struct V6DonnellyStrategyF64;
struct V6DonnellySimdStrategyF64;
struct V6DonnellySimdBlock3StrategyF64;

impl V6StrategySelectorF32 for V6EytzingerStrategyF32 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_mutable_case_f32::<K, B, Eytzinger<K>>(params)
    }
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_immutable_case_f32::<K, B, Eytzinger<K>>(params)
    }
}

impl V6StrategySelectorF32 for V6DonnellyStrategyF32 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_mutable_case_f32::<K, B, Donnelly<4, 64, 4, K>>(params)
    }
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_immutable_case_f32::<K, B, Donnelly<4, 64, 4, K>>(params)
    }
}

impl V6StrategySelectorF32 for V6DonnellySimdStrategyF32 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_mutable_case_f32::<K, B, DonnellyMarkerSimd<Block4, 64, 4, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }

    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_immutable_case_f32::<K, B, DonnellyMarkerSimd<Block4, 64, 4, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }
}

impl V6StrategySelectorF32 for V6DonnellySimdBlock4StrategyF32 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_mutable_case_f32::<K, B, DonnellyMarkerSimd<Block4, 64, 4, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd_block4 requires --features simd".to_string())
        }
    }

    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_immutable_case_f32::<K, B, DonnellyMarkerSimd<Block4, 64, 4, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd_block4 requires --features simd".to_string())
        }
    }
}

impl V6StrategySelectorF64 for V6EytzingerStrategyF64 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_mutable_case_f64::<K, B, Eytzinger<K>>(params)
    }
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_immutable_case_f64::<K, B, Eytzinger<K>>(params)
    }
}

impl V6StrategySelectorF64 for V6DonnellyStrategyF64 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_mutable_case_f64::<K, B, Donnelly<3, 64, 8, K>>(params)
    }
    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        run_v6_immutable_case_f64::<K, B, Donnelly<3, 64, 8, K>>(params)
    }
}

impl V6StrategySelectorF64 for V6DonnellySimdStrategyF64 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_mutable_case_f64::<K, B, DonnellyMarkerSimd<Block3, 64, 8, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }

    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_immutable_case_f64::<K, B, DonnellyMarkerSimd<Block3, 64, 8, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd requires --features simd".to_string())
        }
    }
}

impl V6StrategySelectorF64 for V6DonnellySimdBlock3StrategyF64 {
    fn run_mutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_mutable_case_f64::<K, B, DonnellyMarkerSimd<Block3, 64, 8, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd_block3 requires --features simd".to_string())
        }
    }

    fn run_immutable<const K: usize, const B: usize>(params: &ReproParams) -> Result<(), String> {
        let _ = params;
        #[cfg(feature = "simd")]
        {
            run_v6_immutable_case_f64::<K, B, DonnellyMarkerSimd<Block3, 64, 8, K>>(params)
        }
        #[cfg(not(feature = "simd"))]
        {
            Err("donnelly_simd_block3 requires --features simd".to_string())
        }
    }
}

fn run_immutable_strategy_f32<S>(params: &ReproParams) -> Result<(), String>
where
    S: StrategySelectorF32,
{
    match (params.k, params.b) {
        (2, 16) => S::run::<2, 16>(params),
        (2, 32) => S::run::<2, 32>(params),
        (2, 64) => S::run::<2, 64>(params),
        (3, 16) => S::run::<3, 16>(params),
        (3, 32) => S::run::<3, 32>(params),
        (3, 64) => S::run::<3, 64>(params),
        (4, 16) => S::run::<4, 16>(params),
        (4, 32) => S::run::<4, 32>(params),
        (4, 64) => S::run::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for immutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_immutable_strategy_f64<S>(params: &ReproParams) -> Result<(), String>
where
    S: StrategySelectorF64,
{
    match (params.k, params.b) {
        (2, 16) => S::run::<2, 16>(params),
        (2, 32) => S::run::<2, 32>(params),
        (2, 64) => S::run::<2, 64>(params),
        (3, 16) => S::run::<3, 16>(params),
        (3, 32) => S::run::<3, 32>(params),
        (3, 64) => S::run::<3, 64>(params),
        (4, 16) => S::run::<4, 16>(params),
        (4, 32) => S::run::<4, 32>(params),
        (4, 64) => S::run::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for immutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_v6_mutable_strategy_f32<S>(params: &ReproParams) -> Result<(), String>
where
    S: V6StrategySelectorF32,
{
    match (params.k, params.b) {
        (2, 16) => S::run_mutable::<2, 16>(params),
        (2, 32) => S::run_mutable::<2, 32>(params),
        (2, 64) => S::run_mutable::<2, 64>(params),
        (3, 16) => S::run_mutable::<3, 16>(params),
        (3, 32) => S::run_mutable::<3, 32>(params),
        (3, 64) => S::run_mutable::<3, 64>(params),
        (4, 16) => S::run_mutable::<4, 16>(params),
        (4, 32) => S::run_mutable::<4, 32>(params),
        (4, 64) => S::run_mutable::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for v6 mutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_v6_mutable_strategy_f64<S>(params: &ReproParams) -> Result<(), String>
where
    S: V6StrategySelectorF64,
{
    match (params.k, params.b) {
        (2, 16) => S::run_mutable::<2, 16>(params),
        (2, 32) => S::run_mutable::<2, 32>(params),
        (2, 64) => S::run_mutable::<2, 64>(params),
        (3, 16) => S::run_mutable::<3, 16>(params),
        (3, 32) => S::run_mutable::<3, 32>(params),
        (3, 64) => S::run_mutable::<3, 64>(params),
        (4, 16) => S::run_mutable::<4, 16>(params),
        (4, 32) => S::run_mutable::<4, 32>(params),
        (4, 64) => S::run_mutable::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for v6 mutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_v6_immutable_strategy_f32<S>(params: &ReproParams) -> Result<(), String>
where
    S: V6StrategySelectorF32,
{
    match (params.k, params.b) {
        (2, 16) => S::run_immutable::<2, 16>(params),
        (2, 32) => S::run_immutable::<2, 32>(params),
        (2, 64) => S::run_immutable::<2, 64>(params),
        (3, 16) => S::run_immutable::<3, 16>(params),
        (3, 32) => S::run_immutable::<3, 32>(params),
        (3, 64) => S::run_immutable::<3, 64>(params),
        (4, 16) => S::run_immutable::<4, 16>(params),
        (4, 32) => S::run_immutable::<4, 32>(params),
        (4, 64) => S::run_immutable::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for v6 immutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_v6_immutable_strategy_f64<S>(params: &ReproParams) -> Result<(), String>
where
    S: V6StrategySelectorF64,
{
    match (params.k, params.b) {
        (2, 16) => S::run_immutable::<2, 16>(params),
        (2, 32) => S::run_immutable::<2, 32>(params),
        (2, 64) => S::run_immutable::<2, 64>(params),
        (3, 16) => S::run_immutable::<3, 16>(params),
        (3, 32) => S::run_immutable::<3, 32>(params),
        (3, 64) => S::run_immutable::<3, 64>(params),
        (4, 16) => S::run_immutable::<4, 16>(params),
        (4, 32) => S::run_immutable::<4, 32>(params),
        (4, 64) => S::run_immutable::<4, 64>(params),
        _ => Err(format!(
            "unsupported K/B combination for v6 immutable: K={} B={}",
            params.k, params.b
        )),
    }
}

fn run_mutable_case_f32<const K: usize, const B: usize>(
    params: &ReproParams,
) -> Result<(), String> {
    let cfg = fuzz_config_from_env();
    let points = generate_points_f32::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f32::<K>(params, cfg, point_count);

    let mut tree: KdTree<f32, usize, K, B, u32> = KdTree::with_capacity(point_count);
    for (idx, point) in points.iter().enumerate() {
        tree.add(point, idx);
    }

    run_checks::<f32, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => tree.nearest_one::<SquaredEuclidean>(&query),
            Metric::Manhattan => tree.nearest_one::<Manhattan>(&query),
        },
        |metric, max_qty| {
            Ok(match metric {
                Metric::SquaredEuclidean => tree.nearest_n::<SquaredEuclidean>(&query, max_qty),
                Metric::Manhattan => tree.nearest_n::<Manhattan>(&query, max_qty),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => tree.within_unsorted::<SquaredEuclidean>(&query, radius),
            Metric::Manhattan => tree.within_unsorted::<Manhattan>(&query, radius),
        },
    )
}

fn run_mutable_case_f64<const K: usize, const B: usize>(
    params: &ReproParams,
) -> Result<(), String> {
    let cfg = fuzz_config_from_env();
    let points = generate_points_f64::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f64::<K>(params, cfg, point_count);

    let mut tree: KdTree<f64, usize, K, B, u32> = KdTree::with_capacity(point_count);
    for (idx, point) in points.iter().enumerate() {
        tree.add(point, idx);
    }

    run_checks::<f64, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => tree.nearest_one::<SquaredEuclidean>(&query),
            Metric::Manhattan => tree.nearest_one::<Manhattan>(&query),
        },
        |metric, max_qty| {
            Ok(match metric {
                Metric::SquaredEuclidean => tree.nearest_n::<SquaredEuclidean>(&query, max_qty),
                Metric::Manhattan => tree.nearest_n::<Manhattan>(&query, max_qty),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => tree.within_unsorted::<SquaredEuclidean>(&query, radius),
            Metric::Manhattan => tree.within_unsorted::<Manhattan>(&query, radius),
        },
    )
}

fn run_immutable_case_f32<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f32::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f32::<K>(params, cfg, point_count);

    let tree: ImmutableKdTree<f32, usize, SO, K, B> = ImmutableKdTree::new_from_slice(&points);

    run_checks::<f32, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => tree.nearest_one::<SquaredEuclidean>(&query),
            Metric::Manhattan => tree.nearest_one::<Manhattan>(&query),
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => tree.nearest_n::<SquaredEuclidean>(&query, max_qty),
                Metric::Manhattan => tree.nearest_n::<Manhattan>(&query, max_qty),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => tree.within_unsorted::<SquaredEuclidean>(&query, radius),
            Metric::Manhattan => tree.within_unsorted::<Manhattan>(&query, radius),
        },
    )
}

fn run_immutable_case_f64<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f64::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f64::<K>(params, cfg, point_count);

    let tree: ImmutableKdTree<f64, usize, SO, K, B> = ImmutableKdTree::new_from_slice(&points);

    run_checks::<f64, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => tree.nearest_one::<SquaredEuclidean>(&query),
            Metric::Manhattan => tree.nearest_one::<Manhattan>(&query),
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => tree.nearest_n::<SquaredEuclidean>(&query, max_qty),
                Metric::Manhattan => tree.nearest_n::<Manhattan>(&query, max_qty),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => tree.within_unsorted::<SquaredEuclidean>(&query, radius),
            Metric::Manhattan => tree.within_unsorted::<Manhattan>(&query, radius),
        },
    )
}

fn run_v6_mutable_case_f32<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f32::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f32::<K>(params, cfg, point_count);

    let mut tree: V6KdTree<f32, usize, SO, VecOfArrays<f32, usize, K, B>, K, B> =
        V6KdTree::default();
    for (idx, point) in points.iter().enumerate() {
        tree.add(point, idx);
    }

    run_checks::<f32, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => {
                let (distance, item) = tree.nearest_one::<V6SquaredEuclidean<f32>>(&query);
                NearestNeighbour { distance, item }
            }
            Metric::Manhattan => {
                let (distance, item) = tree.nearest_one::<V6Manhattan<f32>>(&query);
                NearestNeighbour { distance, item }
            }
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => {
                    tree.nearest_n::<V6SquaredEuclidean<f32>>(&query, max_qty, true)
                }
                Metric::Manhattan => tree.nearest_n::<V6Manhattan<f32>>(&query, max_qty, true),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => {
                tree.within_unsorted::<V6SquaredEuclidean<f32>>(&query, radius)
            }
            Metric::Manhattan => tree.within_unsorted::<V6Manhattan<f32>>(&query, radius),
        },
    )
}

fn run_v6_mutable_case_f64<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f64::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f64::<K>(params, cfg, point_count);

    let mut tree: V6KdTree<f64, usize, SO, VecOfArrays<f64, usize, K, B>, K, B> =
        V6KdTree::default();
    for (idx, point) in points.iter().enumerate() {
        tree.add(point, idx);
    }

    run_checks::<f64, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => {
                let (distance, item) = tree.nearest_one::<V6SquaredEuclidean<f64>>(&query);
                NearestNeighbour { distance, item }
            }
            Metric::Manhattan => {
                let (distance, item) = tree.nearest_one::<V6Manhattan<f64>>(&query);
                NearestNeighbour { distance, item }
            }
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => {
                    tree.nearest_n::<V6SquaredEuclidean<f64>>(&query, max_qty, true)
                }
                Metric::Manhattan => tree.nearest_n::<V6Manhattan<f64>>(&query, max_qty, true),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => {
                tree.within_unsorted::<V6SquaredEuclidean<f64>>(&query, radius)
            }
            Metric::Manhattan => tree.within_unsorted::<V6Manhattan<f64>>(&query, radius),
        },
    )
}

fn run_v6_immutable_case_f32<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f32::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f32::<K>(params, cfg, point_count);

    let tree: V6KdTree<f32, usize, SO, FlatVec<f32, usize, K, B>, K, B> =
        V6KdTree::new_from_slice(&points);

    run_checks::<f32, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => {
                let (distance, item) = tree.nearest_one::<V6SquaredEuclidean<f32>>(&query);
                NearestNeighbour { distance, item }
            }
            Metric::Manhattan => {
                let (distance, item) = tree.nearest_one::<V6Manhattan<f32>>(&query);
                NearestNeighbour { distance, item }
            }
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => {
                    tree.nearest_n::<V6SquaredEuclidean<f32>>(&query, max_qty, true)
                }
                Metric::Manhattan => tree.nearest_n::<V6Manhattan<f32>>(&query, max_qty, true),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => {
                tree.within_unsorted::<V6SquaredEuclidean<f32>>(&query, radius)
            }
            Metric::Manhattan => tree.within_unsorted::<V6Manhattan<f32>>(&query, radius),
        },
    )
}

fn run_v6_immutable_case_f64<const K: usize, const B: usize, SO>(
    params: &ReproParams,
) -> Result<(), String>
where
    SO: StemStrategy,
{
    let cfg = fuzz_config_from_env();
    let points = generate_points_f64::<K>(params, cfg)?;
    let point_count = points.len();
    let (query, max_qty, radius_sq, radius_man) =
        generate_query_with_params_f64::<K>(params, cfg, point_count);

    let tree: V6KdTree<f64, usize, SO, FlatVec<f64, usize, K, B>, K, B> =
        V6KdTree::new_from_slice(&points);

    run_checks::<f64, K, _, _, _>(
        params,
        &points,
        &query,
        max_qty,
        radius_sq,
        radius_man,
        |metric| match metric {
            Metric::SquaredEuclidean => {
                let (distance, item) = tree.nearest_one::<V6SquaredEuclidean<f64>>(&query);
                NearestNeighbour { distance, item }
            }
            Metric::Manhattan => {
                let (distance, item) = tree.nearest_one::<V6Manhattan<f64>>(&query);
                NearestNeighbour { distance, item }
            }
        },
        |metric, max_qty| {
            let max_qty = NonZeroUsize::new(max_qty).ok_or("max_qty was zero")?;
            Ok(match metric {
                Metric::SquaredEuclidean => {
                    tree.nearest_n::<V6SquaredEuclidean<f64>>(&query, max_qty, true)
                }
                Metric::Manhattan => tree.nearest_n::<V6Manhattan<f64>>(&query, max_qty, true),
            })
        },
        |metric, radius| match metric {
            Metric::SquaredEuclidean => {
                tree.within_unsorted::<V6SquaredEuclidean<f64>>(&query, radius)
            }
            Metric::Manhattan => tree.within_unsorted::<V6Manhattan<f64>>(&query, radius),
        },
    )
}

#[derive(Clone, Copy)]
enum Metric {
    SquaredEuclidean,
    Manhattan,
}

fn run_checks<A, const K: usize, F1, F2, F3>(
    params: &ReproParams,
    points: &[[A; K]],
    query: &[A; K],
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    nearest_one: F1,
    nearest_n: F2,
    within_unsorted: F3,
) -> Result<(), String>
where
    A: Axis,
    SquaredEuclidean: DistanceMetric<A, K>,
    Manhattan: DistanceMetric<A, K>,
    F1: Fn(Metric) -> NearestNeighbour<A, usize>,
    F2: Fn(Metric, usize) -> Result<Vec<NearestNeighbour<A, usize>>, String>,
    F3: Fn(Metric, A) -> Vec<NearestNeighbour<A, usize>>,
{
    let (mut sq_state, mut man_state) = brute_states(points, query, max_qty, radius_sq, radius_man);

    let result_sq = nearest_one(Metric::SquaredEuclidean);
    check_nearest_one("SquaredEuclidean", result_sq, &sq_state)?;

    let result_man = nearest_one(Metric::Manhattan);
    check_nearest_one("Manhattan", result_man, &man_state)?;

    let mut expected_n_sq = sq_state.take_nearest_n_sorted();
    sort_by_distance_then_index(&mut expected_n_sq);
    let mut result_n_sq: Vec<(A, usize)> = nearest_n(Metric::SquaredEuclidean, max_qty)?
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_distance_then_index(&mut result_n_sq);
    if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
        return Err(format!(
            "nearest_n SquaredEuclidean mismatch: {reason} expected={} got={}",
            format_preview(&expected_n_sq, 8),
            format_preview(&result_n_sq, 8)
        ));
    }

    let mut expected_n_man = man_state.take_nearest_n_sorted();
    sort_by_distance_then_index(&mut expected_n_man);
    let mut result_n_man: Vec<(A, usize)> = nearest_n(Metric::Manhattan, max_qty)?
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_distance_then_index(&mut result_n_man);
    if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
        return Err(format!(
            "nearest_n Manhattan mismatch: {reason} expected={} got={}",
            format_preview(&expected_n_man, 8),
            format_preview(&result_n_man, 8)
        ));
    }

    let expected_within_sq = sq_state.take_within_sorted();
    let mut result_within_sq: Vec<(A, usize)> =
        within_unsorted(Metric::SquaredEuclidean, radius_sq)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
    sort_by_distance_then_index(&mut result_within_sq);
    if result_within_sq != expected_within_sq {
        let mismatch = first_within_mismatch(&expected_within_sq, &result_within_sq)
            .map(|(idx, exp, got)| {
                let exp_dist = SquaredEuclidean::dist(query, &points[exp.1]);
                let got_dist = SquaredEuclidean::dist(query, &points[got.1]);
                format!(
                    " first_mismatch_idx={idx} expected_item={exp:?} expected_dist_calc={exp_dist:?} got_item={got:?} got_dist_calc={got_dist:?} radius={radius_sq:?}",
                )
            })
            .unwrap_or_default();
        return Err(format!(
            "within_unsorted SquaredEuclidean mismatch:{mismatch} expected={} got={}",
            format_preview(&expected_within_sq, 8),
            format_preview(&result_within_sq, 8)
        ));
    }

    let expected_within_man = man_state.take_within_sorted();
    let mut result_within_man: Vec<(A, usize)> = within_unsorted(Metric::Manhattan, radius_man)
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_distance_then_index(&mut result_within_man);
    if result_within_man != expected_within_man {
        let mismatch = first_within_mismatch(&expected_within_man, &result_within_man)
            .map(|(idx, exp, got)| {
                let exp_dist = Manhattan::dist(query, &points[exp.1]);
                let got_dist = Manhattan::dist(query, &points[got.1]);
                format!(
                    " first_mismatch_idx={idx} expected_item={exp:?} expected_dist_calc={exp_dist:?} got_item={got:?} got_dist_calc={got_dist:?} radius={radius_man:?}",
                )
            })
            .unwrap_or_default();
        return Err(format!(
            "within_unsorted Manhattan mismatch:{mismatch} expected={} got={}",
            format_preview(&expected_within_man, 8),
            format_preview(&result_within_man, 8)
        ));
    }

    let leaf = params
        .leaf
        .as_deref()
        .map(|value| format!(" leaf={value}"))
        .unwrap_or_default();
    println!(
        "Repro succeeded for kind={} scalar={} strategy={}{} K={} B={} size={} content_seed={} query_seed={}",
        params.kind,
        params.scalar,
        params.strategy,
        leaf,
        params.k,
        params.b,
        params.size,
        params.content_seed,
        params.query_seed
    );

    Ok(())
}

fn check_nearest_one<A: Axis + std::fmt::Debug>(
    metric: &str,
    result: NearestNeighbour<A, usize>,
    expected: &MetricState<A>,
) -> Result<(), String> {
    if result.distance != expected.best_dist {
        return Err(format!(
            "nearest_one {metric} mismatch: expected_dist={:?} got_dist={:?}",
            expected.best_dist, result.distance
        ));
    }

    if !expected.best_items.contains(&result.item) {
        return Err(format!(
            "nearest_one {metric} mismatch: expected_items={:?} got_item={}",
            expected.best_items, result.item
        ));
    }

    Ok(())
}

fn fuzz_config_from_env() -> FuzzConfig {
    FuzzConfig {
        min_pow: read_env_u32("KIDDO_FUZZ_MIN_POW", DEFAULT_MIN_POW),
        max_pow: read_env_u32("KIDDO_FUZZ_MAX_POW", DEFAULT_MAX_POW),
        perturb_min: read_env_i32("KIDDO_FUZZ_PERTURB_MIN", DEFAULT_PERTURB_MIN),
        perturb_max: read_env_i32("KIDDO_FUZZ_PERTURB_MAX", DEFAULT_PERTURB_MAX),
        max_nearest_n: read_env_usize("KIDDO_FUZZ_MAX_NEAREST_N", DEFAULT_MAX_NEAREST_N),
    }
}

fn read_env_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn read_env_i32(key: &str, default: i32) -> i32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn read_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn generate_points_f32<const K: usize>(
    params: &ReproParams,
    cfg: FuzzConfig,
) -> Result<Vec<[f32; K]>, String> {
    let mut rng_content = StdRng::seed_from_u64(params.content_seed);
    let computed_size = random_point_count(cfg, &mut rng_content);

    if computed_size != params.size {
        return Err(format!(
            "size mismatch: repro id size={} computed_size={} (check fuzz config env vars)",
            params.size, computed_size
        ));
    }

    Ok((0..computed_size)
        .map(|_| random_point_f32::<K>(&mut rng_content))
        .collect())
}

fn generate_points_f64<const K: usize>(
    params: &ReproParams,
    cfg: FuzzConfig,
) -> Result<Vec<[f64; K]>, String> {
    let mut rng_content = StdRng::seed_from_u64(params.content_seed);
    let computed_size = random_point_count(cfg, &mut rng_content);

    if computed_size != params.size {
        return Err(format!(
            "size mismatch: repro id size={} computed_size={} (check fuzz config env vars)",
            params.size, computed_size
        ));
    }

    Ok((0..computed_size)
        .map(|_| random_point_f64::<K>(&mut rng_content))
        .collect())
}

fn generate_query_with_params_f32<const K: usize>(
    params: &ReproParams,
    cfg: FuzzConfig,
    point_count: usize,
) -> ([f32; K], usize, f32, f32) {
    let mut rng_query = StdRng::seed_from_u64(params.query_seed);
    let query = random_point_f32::<K>(&mut rng_query);

    let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);
    let max_qty = rng_query.random_range(1..=max_nearest_n);
    let radius_sq = random_radius_f32::<K>(&mut rng_query);
    let radius_man = random_radius_f32::<K>(&mut rng_query);

    (query, max_qty, radius_sq, radius_man)
}

fn generate_query_with_params_f64<const K: usize>(
    params: &ReproParams,
    cfg: FuzzConfig,
    point_count: usize,
) -> ([f64; K], usize, f64, f64) {
    let mut rng_query = StdRng::seed_from_u64(params.query_seed);
    let query = random_point_f64::<K>(&mut rng_query);

    let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);
    let max_qty = rng_query.random_range(1..=max_nearest_n);
    let radius_sq = random_radius_f64::<K>(&mut rng_query);
    let radius_man = random_radius_f64::<K>(&mut rng_query);

    (query, max_qty, radius_sq, radius_man)
}

fn random_point_f32<const K: usize>(rng: &mut StdRng) -> [f32; K] {
    array::from_fn(|_| rng.random_range(-1.0f32..1.0f32))
}

fn random_point_f64<const K: usize>(rng: &mut StdRng) -> [f64; K] {
    array::from_fn(|_| rng.random_range(-1.0f64..1.0f64))
}

fn random_radius_f32<const K: usize>(rng: &mut StdRng) -> f32 {
    rng.random_range(0.0f32..(0.5f32 * K as f32))
}

fn random_radius_f64<const K: usize>(rng: &mut StdRng) -> f64 {
    rng.random_range(0.0f64..(0.5f64 * K as f64))
}

fn random_point_count(cfg: FuzzConfig, rng: &mut StdRng) -> usize {
    let min_pow = cfg.min_pow.min(cfg.max_pow);
    let max_pow = cfg.min_pow.max(cfg.max_pow);
    let exp = rng.random_range(min_pow..=max_pow);
    let base = 1usize << exp;
    let perturb_min = cfg.perturb_min.min(cfg.perturb_max);
    let perturb_max = cfg.perturb_min.max(cfg.perturb_max);
    let perturb = rng.random_range(perturb_min..=perturb_max);

    if perturb < 0 {
        base.saturating_sub((-perturb) as usize).max(1)
    } else {
        base + perturb as usize
    }
}

fn sort_by_distance_then_index<A: Axis>(items: &mut Vec<(A, usize)>) {
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("NaN distance in sort")
            .then_with(|| a.1.cmp(&b.1))
    });
}

fn compare_nearest_n_sorted<A: Axis + std::fmt::Debug>(
    expected: &[(A, usize)],
    got: &[(A, usize)],
) -> Result<(), String> {
    if expected.len() != got.len() {
        return Err(format!(
            "len mismatch expected={} got={}",
            expected.len(),
            got.len()
        ));
    }
    if expected.is_empty() {
        return Ok(());
    }

    let tail_dist = expected.last().unwrap().0;
    let got_tail = got.last().unwrap().0;
    if got_tail != tail_dist {
        return Err(format!(
            "tail distance mismatch expected={tail_dist:?} got={got_tail:?}"
        ));
    }

    let mut i = 0usize;
    let mut j = 0usize;
    while i < expected.len() && expected[i].0 < tail_dist {
        let dist = expected[i].0;
        let mut exp_items = Vec::new();
        while i < expected.len() && expected[i].0 == dist {
            exp_items.push(expected[i].1);
            i += 1;
        }

        if j < got.len() && got[j].0 < dist {
            return Err(format!(
                "unexpected distance in results {got_dist:?} < expected {dist:?}",
                got_dist = got[j].0
            ));
        }
        if j >= got.len() || got[j].0 != dist {
            return Err(format!("missing distance in results {dist:?}"));
        }

        let mut got_items = Vec::new();
        while j < got.len() && got[j].0 == dist {
            got_items.push(got[j].1);
            j += 1;
        }

        exp_items.sort();
        got_items.sort();
        if exp_items != got_items {
            return Err(format!(
                "tie mismatch at dist {dist:?} expected_items={exp_items:?} got_items={got_items:?}"
            ));
        }
    }

    while j < got.len() && got[j].0 < tail_dist {
        return Err(format!(
            "unexpected distance in results {got_dist:?} < tail {tail_dist:?}",
            got_dist = got[j].0
        ));
    }

    Ok(())
}

fn first_within_mismatch<A: Axis + std::fmt::Debug>(
    expected: &[(A, usize)],
    got: &[(A, usize)],
) -> Option<(usize, (A, usize), (A, usize))> {
    let len = expected.len().min(got.len());
    for idx in 0..len {
        if expected[idx] != got[idx] {
            return Some((idx, expected[idx], got[idx]));
        }
    }
    None
}

fn format_preview<A: std::fmt::Debug>(items: &[(A, usize)], limit: usize) -> String {
    let len = items.len();
    if len <= limit {
        format!("{items:?}")
    } else {
        format!("{:?} ... (len={len})", &items[..limit])
    }
}

struct MetricState<A: Axis> {
    best_dist: A,
    best_items: Vec<usize>,
    heap: BinaryHeap<NearestNeighbour<A, usize>>,
    within: Vec<(A, usize)>,
    max_qty: usize,
    radius: A,
}

impl<A: Axis> MetricState<A> {
    fn new(max_qty: usize, radius: A) -> Self {
        Self {
            best_dist: A::infinity(),
            best_items: Vec::new(),
            heap: BinaryHeap::with_capacity(max_qty + 1),
            within: Vec::new(),
            max_qty,
            radius,
        }
    }

    fn update(&mut self, dist: A, idx: usize) {
        if dist < self.best_dist {
            self.best_dist = dist;
            self.best_items.clear();
            self.best_items.push(idx);
        } else if dist == self.best_dist {
            self.best_items.push(idx);
        }

        if dist <= self.radius {
            self.within.push((dist, idx));
        }

        if self.max_qty == 0 {
            return;
        }

        if self.heap.len() < self.max_qty {
            self.heap.push(NearestNeighbour {
                distance: dist,
                item: idx,
            });
        } else if let Some(top) = self.heap.peek() {
            if dist < top.distance {
                self.heap.pop();
                self.heap.push(NearestNeighbour {
                    distance: dist,
                    item: idx,
                });
            }
        }
    }

    fn take_nearest_n_sorted(&mut self) -> Vec<(A, usize)> {
        let heap = std::mem::take(&mut self.heap);
        heap.into_sorted_vec()
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect()
    }

    fn take_within_sorted(&mut self) -> Vec<(A, usize)> {
        let mut items = std::mem::take(&mut self.within);
        sort_by_distance_then_index(&mut items);
        items
    }
}

fn brute_states<A: Axis, const K: usize>(
    points: &[[A; K]],
    query: &[A; K],
    max_qty: usize,
    radius_sq: A,
    radius_manhattan: A,
) -> (MetricState<A>, MetricState<A>)
where
    SquaredEuclidean: DistanceMetric<A, K>,
    Manhattan: DistanceMetric<A, K>,
{
    let mut sq = MetricState::new(max_qty, radius_sq);
    let mut man = MetricState::new(max_qty, radius_manhattan);

    for (idx, point) in points.iter().enumerate() {
        let dist_sq = SquaredEuclidean::dist(query, point);
        let dist_man = Manhattan::dist(query, point);
        sq.update(dist_sq, idx);
        man.update(dist_man, idx);
    }

    (sq, man)
}
