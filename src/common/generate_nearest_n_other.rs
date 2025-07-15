use kiddo::KdTree;
use std::collections::HashMap;

pub struct MyKdTree<const K: usize> {
    tree: KdTree<f64, K>,
    points: Vec<([f64; K], usize)>,
}

impl<const K: usize> MyKdTree<K> {
    pub fn new() -> Self {
        Self {
            tree: KdTree::new(),
            points: Vec::new(),
        }
    }

    pub fn add(&mut self, point: [f64; K], id: usize) {
        self.tree.add(&point, id);
        self.points.push((point, id));
    }

    pub fn query_ball_tree(
        &self,
        other: &MyKdTree<K>,
        r: f64,
        distance_fn: impl Fn(&[f64; K], &[f64; K]) -> f64,
    ) -> HashMap<usize, Vec<usize>> {
        let mut result = HashMap::new();

        for (self_point, self_id) in &self.points {
            for (other_point, other_id) in &other.points {
                if distance_fn(self_point, other_point) <= r {
                    result.entry(*self_id).or_insert_with(Vec::new).push(*other_id);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kd_tree() {

        let mut tree1: MyKdTree<2> = MyKdTree::new();
        for i in 0..100 {
            let x = i as f64;
            let y = i as f64;
            tree1.add([x, y], i);
        }

        let mut tree2: MyKdTree<2> = MyKdTree::new();
        for i in 0..100 {
            let x = i as f64 + 0.1;
            let y = i as f64 + 0.1;
            tree2.add([x, y], i + 100);
        }

        let distance_fn = |a: &[f64; 2], b: &[f64; 2]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let result = tree1.query_ball_tree(&tree2, 0.15, distance_fn);

        for i in 0..100 {
            let expected_id = i + 100;
            assert_eq!(result.get(&i), Some(&vec![expected_id]));
        }

        assert_eq!(result.len(), 100);

    }
}