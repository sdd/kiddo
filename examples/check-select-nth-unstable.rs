/// Check select_nth_unstable
///
/// This is not really a Kiddo example. The `ImmutableTree` uses `select_nth_unstable` during
/// tree construction and I needed to validate some of the behaviour of that function.
/// Specifically, whether or not **all** items with the same value as the selected item end up adjacent
/// to it in the array or not. It is not clear from the documentation that this is definitely the case.
///
/// If this does happen, we can take advantage of this to increase the
/// performance of construction.
///
/// If this example does not panic when executed, then the condition that we need holds true
/// across every test that we tried.
///
use std::error::Error;
use std::io;
use std::io::Write;

use rand::seq::SliceRandom;
use rand::thread_rng;

const NUM_TRIALS: usize = 1000;
const NUM_DISTINCT_VALS: usize = 1_000_000;
const TIMES_REPEATED: usize = 10;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = thread_rng();

    println!("Starting test\n");
    for trial_num in 0..NUM_TRIALS {
        let progress = trial_num * 100 / NUM_TRIALS;
        print!("\rProgress: {:?}%", progress);
        let _ = io::stdout().flush();

        // 1) Build a list, consisting of the numbers 0 to NUM_DISTINCT_VALS,
        // each duplicated TIMES_REPEATED times
        let mut items = Vec::with_capacity(NUM_DISTINCT_VALS * TIMES_REPEATED);
        for i in 0..NUM_DISTINCT_VALS {
            for _ in 0..TIMES_REPEATED {
                items.push(i);
            }
        }

        // 2) Shuffle the list
        items.shuffle(&mut rng);

        // 3) Perform quickselect to get item (NUM_DISTINCT_VALS / 2) + (TIMES_REPEATED / 2) in the correct place
        let selection_index = (NUM_DISTINCT_VALS * TIMES_REPEATED / 2) + (TIMES_REPEATED / 2);
        items.select_nth_unstable(selection_index);

        // 4) Sanity Check
        let expected_value = NUM_DISTINCT_VALS / 2;
        assert_eq!(items[selection_index], expected_value);

        // 5) Assert that our expectation on the behaviour of select_nth_unstable holds
        let window_min = NUM_DISTINCT_VALS * TIMES_REPEATED / 2;
        for &item in items.iter().take(selection_index).skip(window_min) {
            assert_eq!(item, expected_value);
        }
        assert_ne!(items[window_min - 1], expected_value);
    }
    print!("\rProgress: 100%");

    println!("\n\nSuccess: select_nth_unstable appears to conform to the behaviour we require.");
    Ok(())
}
