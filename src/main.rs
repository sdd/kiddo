use num_traits::Float;
use rand::Rng;
use sok::distance::squared_euclidean;
use sok::KdTree;

fn main() {
    let mut tree: KdTree<f64, i32, 2, 4> = KdTree::new();

    let content_to_add = [
        ([9f64, 0f64], 9),
        ([4f64, 500f64], 4),
        ([12f64, -300f64], 12),
        ([7f64, 200f64], 7),
        ([13f64, -400f64], 13),
        ([6f64, 300f64], 6),
        ([2f64, 700f64], 2),
        ([14f64, -500f64], 14),
        ([3f64, 600f64], 3),
        ([10f64, -100f64], 10),
        ([16f64, -700f64], 16),
        ([1f64, 800f64], 1),
        ([15f64, -600f64], 15),
        ([5f64, 400f64], 5),
        ([8f64, 100f64], 8),
        ([11f64, -200f64], 11),
    ];

    for (point, item) in content_to_add {
        tree.add(&point, item);
    }

    let mut rng = rand::thread_rng();
    for i in 0..1000 {
        let query_point = [
            rng.gen_range(-10f64..20f64),
            rng.gen_range(-1000f64..1000f64),
        ];
        let expected = linear_search(&content_to_add, &query_point);

        let result = tree.nearest_one(&query_point, &squared_euclidean);

        if result != expected {
            println!(
                "Bad: #{:?}. Query: {:?}, Expected: {:?}, Actual: {:?}",
                i, &query_point, &expected, &result
            );
        } else {
            println!("Good: {:?}", i);
        }
    }
}

fn linear_search(content: &[([f64; 2], i32)], query_point: &[f64; 2]) -> (f64, i32) {
    let mut best_dist: f64 = f64::infinity();
    let mut best_item: i32 = i32::MAX;

    for &(p, item) in content {
        let dist = squared_euclidean(query_point, &p);
        if dist < best_dist {
            best_item = item;
            best_dist = dist;
        }
    }

    (best_dist, best_item)
}
