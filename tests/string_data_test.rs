use kiddo::mutable::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;

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

#[test]
fn test_kdtree_with_numeric_id() {
    // Create a new KdTree with 2D points (K=2) and u32 as the data type
    let mut tree: KdTree<f64, u32, 2, 32, u32> = KdTree::new();

    // Add some points with associated numeric IDs
    tree.add(&[0.0, 0.0], 1001);
    tree.add(&[1.0, 1.0], 1002);
    tree.add(&[2.0, 2.0], 1003);
    tree.add(&[-1.0, -1.0], 1004);

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree.nearest_one::<SquaredEuclidean>(&query_point);

    // The closest point should be [0.0, 0.0] with ID 1001
    assert_eq!(nearest.item, 1001);
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);

    // Test k-nearest neighbors
    let k_nearest = tree.nearest_n::<SquaredEuclidean>(&query_point, 2);
    assert_eq!(k_nearest.len(), 2);
    assert_eq!(k_nearest[0].item, 1001);
    assert_eq!(k_nearest[1].item, 1002);

    // Test within radius
    let radius = 2.0;
    let within_results = tree.within::<SquaredEuclidean>(&query_point, radius);
    assert_eq!(within_results.len(), 2); // Should find two points within radius
    assert!(within_results.iter().any(|r| r.item == 1001));
    assert!(within_results.iter().any(|r| r.item == 1002));
}

#[test]
/// Test case with an esoteric T type
fn test_kdtree_with_fixed_string() {
    // Create a new KdTree with 2D points (K=2) and MyFixedString as the data type
    let mut tree: KdTree<f64, MyFixedString, 2, 32, u32> = KdTree::new();

    // Add some points with associated MyFixedString data
    tree.add(&[0.0, 0.0], MyFixedString::from_str("Origin"));
    tree.add(&[1.0, 1.0], MyFixedString::from_str("Point A"));
    tree.add(&[2.0, 2.0], MyFixedString::from_str("Point B"));
    tree.add(&[-1.0, -1.0], MyFixedString::from_str("Point C"));

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree.nearest_one::<SquaredEuclidean>(&query_point);

    // The closest point should be [0.0, 0.0] with data "Origin"
    assert_eq!(nearest.item.as_str(), "Origin");
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);

    let (nearest, nearest_point) = tree.nearest_one_point::<SquaredEuclidean>(&query_point);

    // The closest point should be [0.0, 0.0] with data "Origin"
    assert_eq!(nearest.item.as_str(), "Origin");
    assert_eq!(nearest_point, [0.0, 0.0]);
    assert!((nearest.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);

    // Test k-nearest neighbors
    let k_nearest = tree.nearest_n::<SquaredEuclidean>(&query_point, 2);
    assert_eq!(k_nearest.len(), 2);
    assert_eq!(k_nearest[0].item.as_str(), "Origin");
    assert_eq!(k_nearest[1].item.as_str(), "Point A");

    // Test within radius
    let radius = 2.0;
    let within_results = tree.within::<SquaredEuclidean>(&query_point, radius);
    assert_eq!(within_results.len(), 2); // Should find "Origin" and "Point A"
}

#[test]
fn test_kdtree_with_empty_type_as_content() {
    // Create a new KdTree with 2D points (K=2) and () as the data type
    let mut tree: KdTree<f64, (), 2, 32, u32> = KdTree::new();

    // Add some points with associated numeric IDs
    tree.add(&[0.0, 0.0], ());
    tree.add(&[1.0, 1.0], ());
    tree.add(&[2.0, 2.0], ());
    tree.add(&[-1.0, -1.0], ());

    // Test nearest neighbor query
    let query_point = [0.5, 0.25];
    let nearest = tree.nearest_one_point::<SquaredEuclidean>(&query_point);

    // The closest point should be [0.0, 0.0]
    assert!((nearest.0.distance - (0.5f64 * 0.5 + 0.25 * 0.25)).abs() < f64::EPSILON);
    assert_eq!(nearest.1, [0.0, 0.0]);

    // Test k-nearest neighbors
    let k_nearest = tree.nearest_n::<SquaredEuclidean>(&query_point, 2);
    assert_eq!(k_nearest.len(), 2);

    // Test within radius
    let radius = 2.0;
    let within_results = tree.within::<SquaredEuclidean>(&query_point, radius);
    assert_eq!(within_results.len(), 2); // Should find two points within radius
}
