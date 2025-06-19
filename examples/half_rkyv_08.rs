#[cfg(all(feature = "f16_rkyv_08", not(feature = "f16")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half_2_5::f16;
    use kiddo::{KdTree, SquaredEuclidean};
    use num_traits::real::Real;

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

#[cfg(any(not(feature = "f16_rkyv_08"), feature = "f16"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Activate the 'f16_rkyv_08' feature to run this example properly. Ensure that the 'f16' feature is disabled also");
    println!("Try this: cargo run --example half_rkyv_08 --features=f16_rkyv_08");

    Ok(())
}
