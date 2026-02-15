use std::array;
use std::collections::BinaryHeap;
use std::env;
use std::fs::OpenOptions;
use std::io::{IsTerminal, Write};
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

use kiddo::distance::float::{Manhattan, SquaredEuclidean};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::mutable::float::kdtree::KdTree;
use kiddo::nearest_neighbour::NearestNeighbour;
use kiddo::stem_strategies::{Donnelly, Eytzinger};
use kiddo::traits::{Axis, DistanceMetric};
use kiddo::StemStrategy;

#[cfg(feature = "simd")]
use kiddo::stem_strategies::{Block3, Block4, DonnellyMarkerSimd};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use zzz::ProgressBar;

const DEFAULT_CASES: usize = 5;
const DEFAULT_MIN_POW: u32 = 10;
const DEFAULT_MAX_POW: u32 = 24;
const DEFAULT_PERTURB_MIN: i32 = -5;
const DEFAULT_PERTURB_MAX: i32 = 5;
const DEFAULT_MAX_NEAREST_N: usize = 32;
const DEFAULT_SEED: u64 = 0x4b1d_f00d;
const QUERY_COUNT: usize = 100;
const PROGRESS_EVERY: usize = 100;
const PREVIEW_LEN: usize = 8;
const REPORT_PATH: &str = "kd_tree_fuzz_report.txt";
const SEED_MIX_CASE: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_MIX_QUERY: u64 = 0xbf58_476d_1ce4_e5b9;

static REPORT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

// Long-running fuzz-style correctness checks. Override with env:
// KIDDO_FUZZ_SEED, KIDDO_FUZZ_CASES, KIDDO_FUZZ_MIN_POW, KIDDO_FUZZ_MAX_POW,
// KIDDO_FUZZ_PERTURB_MIN, KIDDO_FUZZ_PERTURB_MAX, KIDDO_FUZZ_MAX_NEAREST_N.
#[derive(Clone, Copy)]
struct FuzzConfig {
    seed: u64,
    cases: usize,
    min_pow: u32,
    max_pow: u32,
    perturb_min: i32,
    perturb_max: i32,
    max_nearest_n: usize,
}

#[derive(Clone, Copy)]
struct ReproMeta {
    kind: &'static str,
    scalar: &'static str,
    strategy: &'static str,
    b: usize,
    k: usize,
}

impl FuzzConfig {
    fn from_env() -> Self {
        Self {
            seed: read_env_u64("KIDDO_FUZZ_SEED", DEFAULT_SEED),
            cases: read_env_usize("KIDDO_FUZZ_CASES", DEFAULT_CASES),
            min_pow: read_env_u32("KIDDO_FUZZ_MIN_POW", DEFAULT_MIN_POW),
            max_pow: read_env_u32("KIDDO_FUZZ_MAX_POW", DEFAULT_MAX_POW),
            perturb_min: read_env_i32("KIDDO_FUZZ_PERTURB_MIN", DEFAULT_PERTURB_MIN),
            perturb_max: read_env_i32("KIDDO_FUZZ_PERTURB_MAX", DEFAULT_PERTURB_MAX),
            max_nearest_n: read_env_usize("KIDDO_FUZZ_MAX_NEAREST_N", DEFAULT_MAX_NEAREST_N),
        }
    }

    fn case_seed(self, case_idx: usize) -> u64 {
        self.seed ^ (case_idx as u64).wrapping_mul(SEED_MIX_CASE)
    }
}

fn read_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
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

fn read_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn log_failure(message: &str) {
    let lock = REPORT_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().ok();

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(REPORT_PATH)
    {
        let _ = writeln!(file, "{message}");
    }
}

fn log_and_panic(message: String) -> ! {
    log_failure(&message);
    panic!("{message}");
}

fn build_repro_id(meta: ReproMeta, size: usize, content_seed: u64, query_seed: u64) -> String {
    format!(
        "failure-kind_{}-ty_{}-strategy_{}-b_{}-k_{}-size_{}-content_seed_{}-query_seed_{}",
        meta.kind, meta.scalar, meta.strategy, meta.b, meta.k, size, content_seed, query_seed
    )
}

fn format_preview<A: std::fmt::Debug>(items: &[(A, usize)], limit: usize) -> String {
    let len = items.len();
    if len <= limit {
        format!("{items:?}")
    } else {
        format!("{:?} ... (len={len})", &items[..limit])
    }
}

fn query_seed(content_seed: u64, query_idx: usize) -> u64 {
    content_seed.wrapping_add(SEED_MIX_QUERY) ^ (query_idx as u64).wrapping_mul(SEED_MIX_CASE)
}

fn random_point_f32<const K: usize>(rng: &mut StdRng) -> [f32; K] {
    array::from_fn(|_| rng.random_range(-1.0f32..1.0f32))
}

fn random_point_f64<const K: usize>(rng: &mut StdRng) -> [f64; K] {
    array::from_fn(|_| rng.random_range(-1.0f64..1.0f64))
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

fn random_radius_f32<const K: usize>(rng: &mut StdRng) -> f32 {
    rng.random_range(0.0f32..(0.5f32 * K as f32))
}

fn random_radius_f64<const K: usize>(rng: &mut StdRng) -> f64 {
    rng.random_range(0.0f64..(0.5f64 * K as f64))
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

fn format_context<A: std::fmt::Display>(
    label: &str,
    case_idx: usize,
    query_idx: usize,
    _content_seed: u64,
    _query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
) -> String {
    format!(
        "label={label} case={} query={} points={} max_qty={} radius_sq={} radius_man={}",
        case_idx + 1,
        query_idx + 1,
        point_count,
        max_qty,
        radius_sq,
        radius_man
    )
}

fn log_mismatch<A: std::fmt::Display>(
    meta: ReproMeta,
    label: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    detail: String,
) -> ! {
    let repro = build_repro_id(meta, point_count, content_seed, query_seed);
    let context = format_context(
        label,
        case_idx,
        query_idx,
        content_seed,
        query_seed,
        point_count,
        max_qty,
        radius_sq,
        radius_man,
    );
    log_and_panic(format!("repro={repro} {context} {detail}"));
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

fn assert_nearest_one<A: Axis + std::fmt::Display>(
    meta: ReproMeta,
    label: &str,
    metric: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    result: NearestNeighbour<A, usize>,
    expected: &MetricState<A>,
) {
    if result.distance != expected.best_dist {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_one mismatch=distance expected={} got={}",
                expected.best_dist, result.distance
            ),
        );
    }

    if !expected.best_items.contains(&result.item) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_one mismatch=item expected_one_of={:?} got={}",
                expected.best_items, result.item
            ),
        );
    }
}

fn print_case_start(label: &str, case_idx: usize, cases: usize, point_count: usize, seed: u64) {
    println!(
        "{label}: case {}/{} with {} points (content_seed={seed})",
        case_idx + 1,
        cases,
        point_count
    );
}

fn print_query_progress(label: &str, case_idx: usize, query_idx: usize, seed: u64) {
    if query_idx.is_multiple_of(PROGRESS_EVERY) {
        println!(
            "{label}: case {} query {}/{} (query_seed={seed})",
            case_idx + 1,
            query_idx + 1,
            QUERY_COUNT
        );
    }
}

struct ProgressReporter {
    bar: Option<ProgressBar>,
    label: String,
    cases: usize,
}

impl ProgressReporter {
    fn new(label: &str, cases: usize) -> Self {
        if std::io::stderr().is_terminal() {
            let total = cases.saturating_mul(QUERY_COUNT).max(1);
            let mut bar = ProgressBar::with_target(total);
            bar.set_message(Some(format!("{label} cases {cases}")));
            Self {
                bar: Some(bar),
                label: label.to_string(),
                cases,
            }
        } else {
            Self {
                bar: None,
                label: label.to_string(),
                cases,
            }
        }
    }

    fn case_start(&self, case_idx: usize, point_count: usize, content_seed: u64) {
        if self.bar.is_none() {
            print_case_start(&self.label, case_idx, self.cases, point_count, content_seed);
        }
    }

    fn advance(&mut self, case_idx: usize, query_idx: usize, _content_seed: u64, query_seed: u64) {
        if let Some(bar) = self.bar.as_mut() {
            if query_idx.is_multiple_of(PROGRESS_EVERY) || query_idx + 1 == QUERY_COUNT {
                bar.set_message(Some(format!(
                    "{} case {}/{} query {}/{} ",
                    self.label,
                    case_idx + 1,
                    self.cases,
                    query_idx + 1,
                    QUERY_COUNT,
                )));
            }
            bar.add(1);
        } else {
            print_query_progress(&self.label, case_idx, query_idx, query_seed);
        }
    }
}

fn run_mutable_case_f32<const K: usize, const B: usize>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) {
    let mut progress = ProgressReporter::new(label, cfg.cases);
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f32; K]> = (0..point_count)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        progress.case_start(case_idx, point_count, content_seed);

        let mut tree: KdTree<f32, usize, K, B, u32> = KdTree::with_capacity(point_count);
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx);
        }

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        for query_idx in 0..QUERY_COUNT {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f32::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f32::<K>(&mut rng_query);
            let radius_man = random_radius_f32::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree.nearest_one::<SquaredEuclidean>(&query);
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.nearest_one::<Manhattan>(&query);
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let mut result_n_sq: Vec<(f32, usize)> = tree
                .nearest_n::<SquaredEuclidean>(&query, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let mut result_n_man: Vec<(f32, usize)> = tree
                .nearest_n::<Manhattan>(&query, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f32, usize)> = tree
                .within_unsorted::<SquaredEuclidean>(&query, radius_sq)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_sq);
            if result_within_sq != expected_within_sq {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f32, usize)> = tree
                .within_unsorted::<Manhattan>(&query, radius_man)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_man);
            if result_within_man != expected_within_man {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            progress.advance(case_idx, query_idx, content_seed, query_seed);
        }
    }
}

fn run_mutable_case_f64<const K: usize, const B: usize>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) {
    let mut progress = ProgressReporter::new(label, cfg.cases);
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f64; K]> = (0..point_count)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        progress.case_start(case_idx, point_count, content_seed);

        let mut tree: KdTree<f64, usize, K, B, u32> = KdTree::with_capacity(point_count);
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx);
        }

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        for query_idx in 0..QUERY_COUNT {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f64::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f64::<K>(&mut rng_query);
            let radius_man = random_radius_f64::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree.nearest_one::<SquaredEuclidean>(&query);
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.nearest_one::<Manhattan>(&query);
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let mut result_n_sq: Vec<(f64, usize)> = tree
                .nearest_n::<SquaredEuclidean>(&query, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let mut result_n_man: Vec<(f64, usize)> = tree
                .nearest_n::<Manhattan>(&query, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f64, usize)> = tree
                .within_unsorted::<SquaredEuclidean>(&query, radius_sq)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_sq);
            if result_within_sq != expected_within_sq {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f64, usize)> = tree
                .within_unsorted::<Manhattan>(&query, radius_man)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_man);
            if result_within_man != expected_within_man {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            progress.advance(case_idx, query_idx, content_seed, query_seed);
        }
    }
}

fn run_immutable_case_f32<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy,
{
    let mut progress = ProgressReporter::new(label, cfg.cases);
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f32; K]> = (0..point_count)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        progress.case_start(case_idx, point_count, content_seed);

        let tree: ImmutableKdTree<f32, usize, SO, K, B> = ImmutableKdTree::new_from_slice(&points);

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        for query_idx in 0..QUERY_COUNT {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f32::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f32::<K>(&mut rng_query);
            let radius_man = random_radius_f32::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree.nearest_one::<SquaredEuclidean>(&query);
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.nearest_one::<Manhattan>(&query);
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f32, usize)> = tree
                .nearest_n::<SquaredEuclidean>(&query, max_qty_nz)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f32, usize)> = tree
                .nearest_n::<Manhattan>(&query, max_qty_nz)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f32, usize)> = tree
                .within_unsorted::<SquaredEuclidean>(&query, radius_sq)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_sq);
            if result_within_sq != expected_within_sq {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f32, usize)> = tree
                .within_unsorted::<Manhattan>(&query, radius_man)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_man);
            if result_within_man != expected_within_man {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            progress.advance(case_idx, query_idx, content_seed, query_seed);
        }
    }
}

fn run_immutable_case_f64<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy,
{
    let mut progress = ProgressReporter::new(label, cfg.cases);
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f64; K]> = (0..point_count)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        progress.case_start(case_idx, point_count, content_seed);

        let tree: ImmutableKdTree<f64, usize, SO, K, B> = ImmutableKdTree::new_from_slice(&points);

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        for query_idx in 0..QUERY_COUNT {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f64::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f64::<K>(&mut rng_query);
            let radius_man = random_radius_f64::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree.nearest_one::<SquaredEuclidean>(&query);
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.nearest_one::<Manhattan>(&query);
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f64, usize)> = tree
                .nearest_n::<SquaredEuclidean>(&query, max_qty_nz)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f64, usize)> = tree
                .nearest_n::<Manhattan>(&query, max_qty_nz)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f64, usize)> = tree
                .within_unsorted::<SquaredEuclidean>(&query, radius_sq)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_sq);
            if result_within_sq != expected_within_sq {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f64, usize)> = tree
                .within_unsorted::<Manhattan>(&query, radius_man)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_distance_then_index(&mut result_within_man);
            if result_within_man != expected_within_man {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            progress.advance(case_idx, query_idx, content_seed, query_seed);
        }
    }
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_mutable_f32() {
    let cfg = FuzzConfig::from_env();
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 16,
        k: 2,
    };
    run_mutable_case_f32::<2, 16>(cfg, "mutable f32 K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 32,
        k: 2,
    };
    run_mutable_case_f32::<2, 32>(cfg, "mutable f32 K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 64,
        k: 2,
    };
    run_mutable_case_f32::<2, 64>(cfg, "mutable f32 K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 16,
        k: 3,
    };
    run_mutable_case_f32::<3, 16>(cfg, "mutable f32 K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 32,
        k: 3,
    };
    run_mutable_case_f32::<3, 32>(cfg, "mutable f32 K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 64,
        k: 3,
    };
    run_mutable_case_f32::<3, 64>(cfg, "mutable f32 K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 16,
        k: 4,
    };
    run_mutable_case_f32::<4, 16>(cfg, "mutable f32 K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 32,
        k: 4,
    };
    run_mutable_case_f32::<4, 32>(cfg, "mutable f32 K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f32",
        strategy: "mutable",
        b: 64,
        k: 4,
    };
    run_mutable_case_f32::<4, 64>(cfg, "mutable f32 K=4 B=64", meta);
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_mutable_f64() {
    let cfg = FuzzConfig::from_env();
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 16,
        k: 2,
    };
    run_mutable_case_f64::<2, 16>(cfg, "mutable f64 K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 32,
        k: 2,
    };
    run_mutable_case_f64::<2, 32>(cfg, "mutable f64 K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 64,
        k: 2,
    };
    run_mutable_case_f64::<2, 64>(cfg, "mutable f64 K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 16,
        k: 3,
    };
    run_mutable_case_f64::<3, 16>(cfg, "mutable f64 K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 32,
        k: 3,
    };
    run_mutable_case_f64::<3, 32>(cfg, "mutable f64 K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 64,
        k: 3,
    };
    run_mutable_case_f64::<3, 64>(cfg, "mutable f64 K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 16,
        k: 4,
    };
    run_mutable_case_f64::<4, 16>(cfg, "mutable f64 K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 32,
        k: 4,
    };
    run_mutable_case_f64::<4, 32>(cfg, "mutable f64 K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "mutable",
        scalar: "f64",
        strategy: "mutable",
        b: 64,
        k: 4,
    };
    run_mutable_case_f64::<4, 64>(cfg, "mutable f64 K=4 B=64", meta);
}

type DonnellyF32<const K: usize> = Donnelly<4, 64, 4, K>;
type DonnellyF64<const K: usize> = Donnelly<3, 64, 8, K>;

#[cfg(feature = "simd")]
type DonnellySimdF32<const K: usize> = DonnellyMarkerSimd<Block4, 64, 4, K>;
#[cfg(feature = "simd")]
type DonnellySimdF64<const K: usize> = DonnellyMarkerSimd<Block3, 64, 8, K>;

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_immutable_f32() {
    let cfg = FuzzConfig::from_env();

    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };
    run_immutable_case_f32::<2, 16, Eytzinger<2>>(cfg, "immutable f32 Eytzinger K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 32,
        k: 2,
    };
    run_immutable_case_f32::<2, 32, Eytzinger<2>>(cfg, "immutable f32 Eytzinger K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 64,
        k: 2,
    };
    run_immutable_case_f32::<2, 64, Eytzinger<2>>(cfg, "immutable f32 Eytzinger K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 16,
        k: 3,
    };
    run_immutable_case_f32::<3, 16, Eytzinger<3>>(cfg, "immutable f32 Eytzinger K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 32,
        k: 3,
    };
    run_immutable_case_f32::<3, 32, Eytzinger<3>>(cfg, "immutable f32 Eytzinger K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 64,
        k: 3,
    };
    run_immutable_case_f32::<3, 64, Eytzinger<3>>(cfg, "immutable f32 Eytzinger K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 16,
        k: 4,
    };
    run_immutable_case_f32::<4, 16, Eytzinger<4>>(cfg, "immutable f32 Eytzinger K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 32,
        k: 4,
    };
    run_immutable_case_f32::<4, 32, Eytzinger<4>>(cfg, "immutable f32 Eytzinger K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "eytzinger",
        b: 64,
        k: 4,
    };
    run_immutable_case_f32::<4, 64, Eytzinger<4>>(cfg, "immutable f32 Eytzinger K=4 B=64", meta);

    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 16,
        k: 2,
    };
    run_immutable_case_f32::<2, 16, DonnellyF32<2>>(cfg, "immutable f32 Donnelly K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 32,
        k: 2,
    };
    run_immutable_case_f32::<2, 32, DonnellyF32<2>>(cfg, "immutable f32 Donnelly K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 64,
        k: 2,
    };
    run_immutable_case_f32::<2, 64, DonnellyF32<2>>(cfg, "immutable f32 Donnelly K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 16,
        k: 3,
    };
    run_immutable_case_f32::<3, 16, DonnellyF32<3>>(cfg, "immutable f32 Donnelly K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 32,
        k: 3,
    };
    run_immutable_case_f32::<3, 32, DonnellyF32<3>>(cfg, "immutable f32 Donnelly K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 64,
        k: 3,
    };
    run_immutable_case_f32::<3, 64, DonnellyF32<3>>(cfg, "immutable f32 Donnelly K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 16,
        k: 4,
    };
    run_immutable_case_f32::<4, 16, DonnellyF32<4>>(cfg, "immutable f32 Donnelly K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 32,
        k: 4,
    };
    run_immutable_case_f32::<4, 32, DonnellyF32<4>>(cfg, "immutable f32 Donnelly K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f32",
        strategy: "donnelly",
        b: 64,
        k: 4,
    };
    run_immutable_case_f32::<4, 64, DonnellyF32<4>>(cfg, "immutable f32 Donnelly K=4 B=64", meta);

    #[cfg(feature = "simd")]
    {
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 16,
            k: 2,
        };
        run_immutable_case_f32::<2, 16, DonnellySimdF32<2>>(
            cfg,
            "immutable f32 DonnellySimd K=2 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 32,
            k: 2,
        };
        run_immutable_case_f32::<2, 32, DonnellySimdF32<2>>(
            cfg,
            "immutable f32 DonnellySimd K=2 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 64,
            k: 2,
        };
        run_immutable_case_f32::<2, 64, DonnellySimdF32<2>>(
            cfg,
            "immutable f32 DonnellySimd K=2 B=64",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 16,
            k: 3,
        };
        run_immutable_case_f32::<3, 16, DonnellySimdF32<3>>(
            cfg,
            "immutable f32 DonnellySimd K=3 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 32,
            k: 3,
        };
        run_immutable_case_f32::<3, 32, DonnellySimdF32<3>>(
            cfg,
            "immutable f32 DonnellySimd K=3 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 64,
            k: 3,
        };
        run_immutable_case_f32::<3, 64, DonnellySimdF32<3>>(
            cfg,
            "immutable f32 DonnellySimd K=3 B=64",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 16,
            k: 4,
        };
        run_immutable_case_f32::<4, 16, DonnellySimdF32<4>>(
            cfg,
            "immutable f32 DonnellySimd K=4 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 32,
            k: 4,
        };
        run_immutable_case_f32::<4, 32, DonnellySimdF32<4>>(
            cfg,
            "immutable f32 DonnellySimd K=4 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f32",
            strategy: "donnelly_simd",
            b: 64,
            k: 4,
        };
        run_immutable_case_f32::<4, 64, DonnellySimdF32<4>>(
            cfg,
            "immutable f32 DonnellySimd K=4 B=64",
            meta,
        );
    }
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_immutable_f64() {
    let cfg = FuzzConfig::from_env();

    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };
    run_immutable_case_f64::<2, 16, Eytzinger<2>>(cfg, "immutable f64 Eytzinger K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 32,
        k: 2,
    };
    run_immutable_case_f64::<2, 32, Eytzinger<2>>(cfg, "immutable f64 Eytzinger K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 64,
        k: 2,
    };
    run_immutable_case_f64::<2, 64, Eytzinger<2>>(cfg, "immutable f64 Eytzinger K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 16,
        k: 3,
    };
    run_immutable_case_f64::<3, 16, Eytzinger<3>>(cfg, "immutable f64 Eytzinger K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 32,
        k: 3,
    };
    run_immutable_case_f64::<3, 32, Eytzinger<3>>(cfg, "immutable f64 Eytzinger K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 64,
        k: 3,
    };
    run_immutable_case_f64::<3, 64, Eytzinger<3>>(cfg, "immutable f64 Eytzinger K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 16,
        k: 4,
    };
    run_immutable_case_f64::<4, 16, Eytzinger<4>>(cfg, "immutable f64 Eytzinger K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 32,
        k: 4,
    };
    run_immutable_case_f64::<4, 32, Eytzinger<4>>(cfg, "immutable f64 Eytzinger K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "eytzinger",
        b: 64,
        k: 4,
    };
    run_immutable_case_f64::<4, 64, Eytzinger<4>>(cfg, "immutable f64 Eytzinger K=4 B=64", meta);

    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 16,
        k: 2,
    };
    run_immutable_case_f64::<2, 16, DonnellyF64<2>>(cfg, "immutable f64 Donnelly K=2 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 32,
        k: 2,
    };
    run_immutable_case_f64::<2, 32, DonnellyF64<2>>(cfg, "immutable f64 Donnelly K=2 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 64,
        k: 2,
    };
    run_immutable_case_f64::<2, 64, DonnellyF64<2>>(cfg, "immutable f64 Donnelly K=2 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 16,
        k: 3,
    };
    run_immutable_case_f64::<3, 16, DonnellyF64<3>>(cfg, "immutable f64 Donnelly K=3 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 32,
        k: 3,
    };
    run_immutable_case_f64::<3, 32, DonnellyF64<3>>(cfg, "immutable f64 Donnelly K=3 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 64,
        k: 3,
    };
    run_immutable_case_f64::<3, 64, DonnellyF64<3>>(cfg, "immutable f64 Donnelly K=3 B=64", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 16,
        k: 4,
    };
    run_immutable_case_f64::<4, 16, DonnellyF64<4>>(cfg, "immutable f64 Donnelly K=4 B=16", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 32,
        k: 4,
    };
    run_immutable_case_f64::<4, 32, DonnellyF64<4>>(cfg, "immutable f64 Donnelly K=4 B=32", meta);
    let meta = ReproMeta {
        kind: "immutable",
        scalar: "f64",
        strategy: "donnelly",
        b: 64,
        k: 4,
    };
    run_immutable_case_f64::<4, 64, DonnellyF64<4>>(cfg, "immutable f64 Donnelly K=4 B=64", meta);

    #[cfg(feature = "simd")]
    {
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 16,
            k: 2,
        };
        run_immutable_case_f64::<2, 16, DonnellySimdF64<2>>(
            cfg,
            "immutable f64 DonnellySimd K=2 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 32,
            k: 2,
        };
        run_immutable_case_f64::<2, 32, DonnellySimdF64<2>>(
            cfg,
            "immutable f64 DonnellySimd K=2 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 64,
            k: 2,
        };
        run_immutable_case_f64::<2, 64, DonnellySimdF64<2>>(
            cfg,
            "immutable f64 DonnellySimd K=2 B=64",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 16,
            k: 3,
        };
        run_immutable_case_f64::<3, 16, DonnellySimdF64<3>>(
            cfg,
            "immutable f64 DonnellySimd K=3 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 32,
            k: 3,
        };
        run_immutable_case_f64::<3, 32, DonnellySimdF64<3>>(
            cfg,
            "immutable f64 DonnellySimd K=3 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 64,
            k: 3,
        };
        run_immutable_case_f64::<3, 64, DonnellySimdF64<3>>(
            cfg,
            "immutable f64 DonnellySimd K=3 B=64",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 16,
            k: 4,
        };
        run_immutable_case_f64::<4, 16, DonnellySimdF64<4>>(
            cfg,
            "immutable f64 DonnellySimd K=4 B=16",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 32,
            k: 4,
        };
        run_immutable_case_f64::<4, 32, DonnellySimdF64<4>>(
            cfg,
            "immutable f64 DonnellySimd K=4 B=32",
            meta,
        );
        let meta = ReproMeta {
            kind: "immutable",
            scalar: "f64",
            strategy: "donnelly_simd",
            b: 64,
            k: 4,
        };
        run_immutable_case_f64::<4, 64, DonnellySimdF64<4>>(
            cfg,
            "immutable f64 DonnellySimd K=4 B=64",
            meta,
        );
    }
}
