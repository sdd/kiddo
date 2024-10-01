#[cfg(feature = "f16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use kiddo::{KdTree, SquaredEuclidean};
    use num_traits::real::Real;

    // build and serialize small tree for ArchivedKdTree doctests
    let mut tree: KdTree<f16, 3> = KdTree::new();
    tree.add(
        &[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(5.0)],
        100,
    );
    tree.add(
        &[f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(6.0)],
        101,
    );

    let nearest = tree.nearest_one::<SquaredEuclidean>(&[
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(5.1),
    ]);

    println!("Nearest: {:?}", &nearest);

    assert!((nearest.distance - f16::from_f32(0.01)).abs() < f16::EPSILON);
    assert_eq!(nearest.item, 100);
    Ok(())
}

#[cfg(not(feature = "f16"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Activate the 'half' feature to run this example properly");
    println!("Try this: cargo run --example half --features=half");

    Ok(())
}
