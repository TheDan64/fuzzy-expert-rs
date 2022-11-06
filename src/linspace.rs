pub struct Linspace {
    start: f32,
    step: f32,
    index: usize,
    len: usize,
}

impl Linspace {
    pub fn new(min: f32, max: f32, n: usize) -> Self {
        let step = if n > 1 {
            // REVIEW: try_from instead of cast?
            let num_steps = (n - 1) as f32;
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
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            // REVIEW: try_from instead of cast?
            Some(self.start + self.step * i as f32)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}
