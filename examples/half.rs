#[cfg(feature = "f16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use kiddo::dist::SquaredEuclidean;
    use kiddo::kd_tree::leaf_strategies::VecOfArrays;
    use kiddo::kd_tree::KdTree;
    use kiddo::Eytzinger;
    use num_traits::real::Real;

    type Tree = KdTree<f16, u32, Eytzinger<3>, VecOfArrays<f16, u32, 3, 32>, 3, 32>;
    let mut tree: Tree = KdTree::default();
    tree.add(
        &[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(5.0)],
        100,
    );
    tree.add(
        &[f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(6.0)],
        101,
    );

    let result = tree
        .query(&[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(5.1),
        ])
        .nearest_one::<SquaredEuclidean<f16>>()
        .execute();

    println!(
        "Nearest: distance={:?} item={}",
        result.distance, result.item
    );

    assert!((result.distance - f16::from_f32(0.01)).abs() < f16::EPSILON);
    assert_eq!(result.item, 100);
    Ok(())
}

#[cfg(not(feature = "f16"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Activate the 'f16' feature to run this example properly.");
    println!("Try this: cargo run --example half --features=f16");

    Ok(())
}
