use std::fmt::{self, Display};
use std::num::NonZero;

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArrays;
use kiddo::{Eytzinger, SquaredEuclidean};

type TestTree<T> = KdTree<f64, T, Eytzinger, VecOfArrays<f64, T, 2, 32>, 2, 32>;

#[derive(Clone, Copy, Debug, PartialEq, Default, Eq, Ord, PartialOrd)]
struct MyFixedString {
    bytes: [u8; 32],
}

impl MyFixedString {
    fn from_str(s: &str) -> Self {
        let mut bytes = [0u8; 32];
        let copy_length = s.len().min(32);
        bytes[..copy_length].copy_from_slice(&s.as_bytes()[..copy_length]);
        Self { bytes }
    }

    fn as_str(&self) -> &str {
        let len = self.bytes.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&self.bytes[..len]).unwrap()
    }
}

impl Display for MyFixedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[test]
fn test_kdtree_with_numeric_id() {
    // Create a new KdTree with 2D points (K=2) and u32 as the data type
    let mut tree: TestTree<u32> = KdTree::default();

    // Add some points with associated numeric IDs
    tree.add(&[0.0, 0.0], 1001).unwrap();
    tree.add(&[1.0, 1.0], 1002).unwrap();
    tree.add(&[2.0, 2.0], 1003).unwrap();
    tree.add(&[-1.0, -1.0], 1004).unwrap();

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree
        .query(&query_point)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();

    // The closest point should be [0.0, 0.0] with ID 1001
    assert_eq!(nearest.item, 1001);
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);

    // Test k-nearest neighbors
    let k_nearest = tree
        .query(&query_point)
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(2).unwrap())
        .execute();
    assert_eq!(k_nearest.len(), 2);
    assert_eq!(k_nearest[0].item, 1001);
    assert_eq!(k_nearest[1].item, 1002);

    // Test within radius
    let radius = 2.0;
    let within_results = tree
        .query(&query_point)
        .within::<SquaredEuclidean<f64>>(radius)
        .execute();
    assert_eq!(within_results.len(), 2); // Should find two points within radius
    assert!(within_results.iter().any(|r| r.item == 1001));
    assert!(within_results.iter().any(|r| r.item == 1002));
}

#[test]
/// Test case with an esoteric T type
fn test_kdtree_with_fixed_string() {
    // Create a new KdTree with 2D points (K=2) and MyFixedString as the data type
    let mut tree: TestTree<MyFixedString> = KdTree::default();

    // Add some points with associated MyFixedString data
    tree.add(&[0.0, 0.0], MyFixedString::from_str("Origin"))
        .unwrap();
    tree.add(&[1.0, 1.0], MyFixedString::from_str("Point A"))
        .unwrap();
    tree.add(&[2.0, 2.0], MyFixedString::from_str("Point B"))
        .unwrap();
    tree.add(&[-1.0, -1.0], MyFixedString::from_str("Point C"))
        .unwrap();

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree
        .query(&query_point)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();

    // The closest point should be [0.0, 0.0] with data "Origin"
    assert_eq!(nearest.item.as_str(), "Origin");
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);

    // Test k-nearest neighbors
    let k_nearest = tree
        .query(&query_point)
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(2).unwrap())
        .execute();
    assert_eq!(k_nearest.len(), 2);
    assert_eq!(k_nearest[0].item.as_str(), "Origin");
    assert_eq!(k_nearest[1].item.as_str(), "Point A");

    // Test within radius
    let radius = 2.0;
    let within_results = tree
        .query(&query_point)
        .within::<SquaredEuclidean<f64>>(radius)
        .execute();
    assert_eq!(within_results.len(), 2); // Should find "Origin" and "Point A"
}

#[test]
fn test_kdtree_with_empty_type_as_content() {
    // Create a new KdTree with 2D points (K=2) and () as the data type
    let mut tree: TestTree<()> = KdTree::default();

    // Add some points with associated numeric IDs
    tree.add(&[0.0, 0.0], ()).unwrap();
    tree.add(&[1.0, 1.0], ()).unwrap();
    tree.add(&[2.0, 2.0], ()).unwrap();
    tree.add(&[-1.0, -1.0], ()).unwrap();

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree
        .query(&query_point)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();

    // The closest point should be [0.0, 0.0]
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);
    assert_eq!(nearest.item, ());

    // Test k-nearest neighbors
    let k_nearest = tree
        .query(&query_point)
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(2).unwrap())
        .execute();
    assert_eq!(k_nearest.len(), 2);

    // Test within radius
    let radius = 2.0;
    let within_results = tree
        .query(&query_point)
        .within::<SquaredEuclidean<f64>>(radius)
        .execute();
    assert_eq!(within_results.len(), 2); // Should find two points within radius
}
