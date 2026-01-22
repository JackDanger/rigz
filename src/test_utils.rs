#[macro_export]
macro_rules! assert_slices_eq {
    ($left:expr, $right:expr) => {
        let left = &$left[..];
        let right = &$right[..];
        if left != right {
            if left.len() != right.len() {
                panic!(
                    "assertion failed: `(left == right)`\n  left len: {},\n right len: {}",
                    left.len(),
                    right.len()
                );
            }
            for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
                if a != b {
                    let start = i.saturating_sub(16);
                    let end = (i + 16).min(left.len());
                    panic!(
                        "assertion failed: `(left == right)` at index {}\n  left[{:?}]: {:02X?}\n right[{:?}]: {:02X?}\n context around index {}:\n left:  {:02X?}\n right: {:02X?}",
                        i, i, a, i, b, i, &left[start..end], &right[start..end]
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $msg:expr) => {
        let left = &$left[..];
        let right = &$right[..];
        if left != right {
            if left.len() != right.len() {
                panic!(
                    "assertion failed: `(left == right)`: {}\n  left len: {},\n right len: {}",
                    $msg,
                    left.len(),
                    right.len()
                );
            }
            for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
                if a != b {
                    let start = i.saturating_sub(16);
                    let end = (i + 16).min(left.len());
                    panic!(
                        "assertion failed: `(left == right)`: {}\n at index {}\n  left[{:?}]: {:02X?}\n right[{:?}]: {:02X?}\n context around index {}:\n left:  {:02X?}\n right: {:02X?}",
                        $msg, i, i, a, i, b, i, &left[start..end], &right[start..end]
                    );
                }
            }
        }
    };
}
