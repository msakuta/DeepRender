use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Matrix {
    rows: usize,
    cols: usize,
    v: Vec<f64>,
}

impl Matrix {
    pub(crate) fn new<const R: usize, const C: usize>(m: [[f64; C]; R]) -> Self {
        Self {
            rows: R,
            cols: C,
            v: m.into_iter().flatten().collect(),
        }
    }

    pub(crate) fn from_slice<const C: usize>(m: &[[f64; C]]) -> Self {
        Self {
            rows: m.len(),
            cols: C,
            v: m.iter().flatten().copied().collect(),
        }
    }

    pub(crate) fn zeros(row: usize, col: usize) -> Self {
        Self {
            rows: row,
            cols: col,
            v: vec![0.; row * col],
        }
    }

    #[allow(dead_code)]
    pub(crate) fn zeros_like(other: &Matrix) -> Self {
        Self {
            rows: other.rows,
            cols: other.cols,
            v: vec![0.; other.rows * other.cols],
        }
    }

    pub(crate) fn ones(row: usize, col: usize) -> Self {
        Self {
            rows: row,
            cols: col,
            v: vec![1.; row * col],
        }
    }

    pub(crate) fn rand(row: usize, col: usize) -> Self {
        Self {
            rows: row,
            cols: col,
            v: (0..row * col)
                .map(|_| rand::random::<f64>() * 2. - 1.)
                .collect(),
        }
    }

    pub(crate) fn new_row(row: &[f64]) -> Self {
        Self {
            rows: 1,
            cols: row.len(),
            v: row.to_vec(),
        }
    }

    pub(crate) fn hstack(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        let rows = self.rows;
        let cols = self.cols + other.cols;
        let mut v = vec![0.; self.rows * cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                v[r * cols + c] = self.v[r * self.cols + c];
            }
            for c in 0..other.cols {
                v[r * cols + c + self.cols] = other.v[r * other.cols + c];
            }
        }
        Self { rows, cols, v }
    }

    #[allow(dead_code)]
    pub(crate) fn eye(size: usize) -> Self {
        let mut v = vec![0.; size * size];
        for i in 0..size {
            v[i * size + i] = 1.;
        }
        Self {
            rows: size,
            cols: size,
            v,
        }
    }

    pub(crate) fn t(&self) -> Self {
        let mut v = vec![0.; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                v[c * self.rows + r] = self.v[r * self.cols + c];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            v,
        }
    }

    pub(crate) fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            v: self.v.iter().copied().map(f).collect(),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    #[allow(dead_code)]
    pub(crate) fn row_range(&self, start: usize, end: usize) -> Matrix {
        let rows = end - start;
        let mut v = vec![0.; self.cols * rows];
        for r in start..end {
            for c in 0..self.cols {
                v[(r - start) * self.cols + c] = self.v[r * self.cols + c];
            }
        }
        Self {
            rows,
            cols: self.cols,
            v,
        }
    }

    pub(crate) fn row(&self, r: usize) -> &[f64] {
        &self.v[r * self.cols..(r + 1) * self.cols]
    }

    pub(crate) fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.v[r * self.cols..(r + 1) * self.cols]
    }

    pub(crate) fn cols_range(&self, start: usize, end: usize) -> Matrix {
        let cols = end - start;
        let mut v = vec![0.; cols * self.rows];
        for r in 0..self.rows {
            for c in start..end {
                v[r * cols + c] = self.v[r * self.cols + c];
            }
        }
        Self {
            rows: self.rows,
            cols,
            v,
        }
    }

    pub(crate) fn col(&self, c: usize) -> Matrix {
        let mut v = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            v.push(self.v[r * self.cols + c]);
        }
        Self {
            rows: self.rows,
            cols: 1,
            v,
        }
    }

    pub(crate) fn iter_rows(&self) -> impl Iterator<Item = &[f64]> {
        (0..self.rows).map(|r| &self.v[r * self.cols..(r + 1) * self.cols])
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[allow(dead_code)]
    pub(crate) fn sum(&self) -> f64 {
        self.v.iter().fold(0., |acc, cur| acc + *cur)
    }

    #[allow(dead_code)]
    pub(crate) fn sum_col(&self) -> Matrix {
        let mut v = vec![0.; self.cols];
        for c in 0..self.cols {
            for r in 0..self.rows {
                v[c] += self[(r, c)];
            }
        }
        Self {
            rows: 1,
            cols: self.cols,
            v,
        }
    }

    pub(crate) fn flat(&self) -> &[f64] {
        &self.v
    }

    pub(crate) fn scale(&self, f: f64) -> Matrix {
        Self {
            rows: self.rows,
            cols: self.cols,
            v: self.v.iter().map(|v| f * *v).collect(),
        }
    }

    pub(crate) fn zip_mut<'a>(
        &'a mut self,
        other: &'a Matrix,
    ) -> impl Iterator<Item = (&'a mut f64, &'a f64)> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.v.iter_mut().zip(other.v.iter())
    }

    pub(crate) fn elementwise_mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.rows, rhs.rows);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            v: self
                .v
                .iter()
                .zip(rhs.v.iter())
                .map(|(l, r)| *l * *r)
                .collect(),
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.v[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.v[r * self.cols + c]
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for r in 0..self.rows {
            if r != 0 {
                write!(f, "  ")?;
            }
            for c in 0..self.cols {
                write!(f, "{}", self.v[r * self.cols + c])?;
                if c != self.cols - 1 {
                    write!(f, " ")?;
                }
            }
            if r != self.rows - 1 {
                write!(f, "\n")?;
            }
        }
        writeln!(f, " ]")?;
        Ok(())
    }
}

impl std::ops::Add for Matrix {
    type Output = Self;
    fn add(self, other: Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = self.v;
        for (v, o) in v.iter_mut().zip(other.v.iter()) {
            *v += o;
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            v,
        }
    }
}

#[test]
fn test_add() {
    let a = Matrix::eye(3);
    let b = Matrix::eye(3);
    let c = a + b;
    assert_eq!(c, Matrix::new([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]));
}

impl std::ops::AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        for (lhs, rhs) in self.v.iter_mut().zip(rhs.v.iter()) {
            *lhs += *rhs;
        }
    }
}

impl std::ops::Sub for Matrix {
    type Output = Self;
    fn sub(self, other: Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = self.v;
        for (v, o) in v.iter_mut().zip(other.v.iter()) {
            *v -= o;
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            v,
        }
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);
        let mut v = vec![0.; self.rows * rhs.cols];
        for r in 0..self.rows {
            for c in 0..rhs.cols {
                v[r * rhs.cols + c] = self.v[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .zip(0..self.cols)
                    .map(|(s, oi)| s * rhs.v[oi * rhs.cols + c])
                    .fold(0., |acc, cur| acc + cur)
            }
        }
        Matrix {
            rows: self.rows,
            cols: rhs.cols,
            v,
        }
    }
}

#[test]
fn test_mul() {
    let a = Matrix::eye(3);
    let b = Matrix::eye(3);
    let d = &a * &b;
    assert_eq!(d, Matrix::eye(3));

    let aa = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
    let bb = Matrix::new([[1., 2.], [0., 1.], [0., 2.]]);
    let cc = &aa * &bb;
    assert_eq!(cc, Matrix::new([[1., 10.], [4., 25.]]));
}
