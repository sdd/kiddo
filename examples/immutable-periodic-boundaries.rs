use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

fn main() {
    let points = vec![[0.95, 0.50], [0.92, 0.55], [0.40, 0.50], [0.10, 0.10]];
    let tree: ImmutableKdTree<f64, u32, 2, 8> = ImmutableKdTree::new_from_slice(&points);

    let query = [0.05, 0.50];
    let box_size = [1.0, 1.0];
    let radius = 0.03;

    let nearest = tree.nearest_one_periodic::<SquaredEuclidean>(&query, &box_size);
    println!("nearest_one_periodic -> {:?}", nearest);

    let nearest_n = tree.nearest_n_periodic::<SquaredEuclidean>(
        &query,
        std::num::NonZero::new(2).unwrap(),
        &box_size,
    );
    println!("nearest_n_periodic -> {:?}", nearest_n);

    let within = tree.within_periodic::<SquaredEuclidean>(&query, radius, &box_size);
    println!("within_periodic -> {:?}", within);

    let within_unsorted =
        tree.within_unsorted_periodic::<SquaredEuclidean>(&query, radius, &box_size);
    println!("within_unsorted_periodic -> {:?}", within_unsorted);

    let nearest_n_within = tree.nearest_n_within_periodic::<SquaredEuclidean>(
        &query,
        radius,
        std::num::NonZero::new(2).unwrap(),
        true,
        &box_size,
    );
    println!("nearest_n_within_periodic -> {:?}", nearest_n_within);
}
