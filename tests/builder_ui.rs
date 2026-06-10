#[test]
fn periodic_queries_cannot_return_points() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/periodic_with_points.rs");
}
