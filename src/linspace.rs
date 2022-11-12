pub struct Linspace {
    start: f64,
    step: f64,
    index: usize,
    len: usize,
}

impl Linspace {
    pub fn new(min: f64, max: f64, n: usize) -> Self {
        let step = if n > 1 {
            // REVIEW: try_from instead of cast?
            let num_steps = (n - 1) as f64;
            (max - min) / num_steps
        } else {
            0.
        };
        Linspace {
            start: min,
            step,
            index: 0,
            len: n,
        }
    }
}

impl Iterator for Linspace {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<f64> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            // REVIEW: try_from instead of cast?
            Some(self.start + self.step * i as f64)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}
