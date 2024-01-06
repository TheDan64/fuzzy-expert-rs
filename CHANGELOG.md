# 0.2.0 (Unreleased)

- `Variables::add` now takes an optional step value for calculating the universe and defaults to the old value (`0.1`).
- `Variables::add` now returns a result with a new `UnsortableFloatsError` error type if sorting fails due to unsortable values.

# 0.1.0 (January 08, 2023)

- Initial crate released with support for crisp variables.
