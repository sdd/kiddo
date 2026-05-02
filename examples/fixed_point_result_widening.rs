/// Kiddo example 1: Palettization
///
/// This example shows how to create a Kiddo `KdTree` that stores 3D fixed point numbers,
/// and how to query this with a widening accumulator so that squared Euclidean distances
/// can be computed without overflowing.
///
/// These techniques are used to store a 256-colour palette within the tree, which is then
/// used to find the nearest matching colour in the palette for each true-colour pixel in
/// a source image, so that a palettized image can be created from it.
///
use std::error::Error;
use std::fs::File;
use std::io::Write;

use rand::seq::SliceRandom;

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::EytzingerPf;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};

type PaletteTree = KdTree<FixedU8<U0>, u8, EytzingerPf<3, 8>, VecOfArenas<FixedU8<U0>, u8, 3, 32>, 3, 32>;

fn main() -> Result<(), Box<dyn Error>> {

    // load the source image

    // calculate the palette

    // quantize the image

    // save to the target filename

    Ok(())
}

/// Run k-means in RGB space to produce a `k`-color palette.
fn kmeans_palette(
    pixels: &[[u8; 3]],
    k: usize,
    max_iters: usize,
) -> Vec<[u8; 3]> {
    // Work in f64 so centroid averaging doesn't accumulate rounding errors.
    let samples: Vec<[f64; 3]> = pixels
        .iter()
        .map(|&[r, g, b]| [r as f64, g as f64, b as f64])
        .collect();

    // --- k-means++ initialization ---
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(k);
    centroids.push(*samples.choose(&mut rng).unwrap());

    for _ in 1..k {
        // For each sample, distance² to its nearest existing centroid.
        let tree = KdTree::build(&centroids);
        let weights: Vec<f64> = samples
            .iter()
            .map(|p| {
                let (_, dist_sq) = tree.nearest(p);
                dist_sq
            })
            .collect();

        // Weighted random selection — pick points far from existing centroids.
        let total: f64 = weights.iter().sum();
        let mut threshold = rand::random::<f64>() * total;
        let mut chosen = &samples[0];
        for (s, &w) in samples.iter().zip(&weights) {
            threshold -= w;
            if threshold <= 0.0 {
                chosen = s;
                break;
            }
        }
        centroids.push(*chosen);
    }

    // --- Main k-means loop ---
    let mut assignments = vec![0usize; samples.len()];

    for _iter in 0..max_iters {
        // Build a kd-tree over the current centroids.
        let tree = KdTree::build(&centroids);

        // Assign each pixel to its nearest centroid.
        let mut changed = false;
        for (i, pixel) in samples.iter().enumerate() {
            let (nearest_idx, _dist_sq) = tree.nearest(pixel);
            if assignments[i] != nearest_idx {
                assignments[i] = nearest_idx;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids as the mean of their assigned pixels.
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u64; k];

        for (i, pixel) in samples.iter().enumerate() {
            let c = assignments[i];
            sums[c][0] += pixel[0];
            sums[c][1] += pixel[1];
            sums[c][2] += pixel[2];
            counts[c] += 1;
        }

        for (j, centroid) in centroids.iter_mut().enumerate() {
            if counts[j] > 0 {
                let n = counts[j] as f64;
                *centroid = [sums[j][0] / n, sums[j][1] / n, sums[j][2] / n];
            }
            // Empty cluster: leave centroid where it was. It'll either
            // pick up stragglers or stay dead — both are fine for a palette.
        }
    }

    // Snap back to u8.
    centroids
        .iter()
        .map(|&[r, g, b]| [r.round() as u8, g.round() as u8, b.round() as u8])
        .collect()
}

/// Quantize an image to the given palette using the kd-tree for lookup.
fn quantize(pixels: &[[u8; 3]], palette: &[[u8; 3]]) -> Vec<u8> {
    // Build the kd-tree once over the 256 palette entries.
    let palette_f64: Vec<[f64; 3]> = palette
        .iter()
        .map(|&[r, g, b]| [r as f64, g as f64, b as f64])
        .collect();
    let tree = KdTree::build(&palette_f64);

    // Map every pixel to its nearest palette index.
    pixels
        .iter()
        .map(|&[r, g, b]| {
            let query = [r as f64, g as f64, b as f64];
            let (idx, _) = tree.nearest(&query);
            idx as u8
        })
        .collect()
}
