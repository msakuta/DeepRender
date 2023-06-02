use std::{fmt::Display, ops::{Index, IndexMut}};


#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Matrix {
    row: usize,
    col: usize,
    v: Vec<f64>,
}

impl Matrix {
    pub(crate) fn new<const R: usize, const C: usize>(m: [[f64; C]; R]) -> Self {
        Self {
            row: R,
            col: C,
            v: m.into_iter().flatten().collect(),
        }
    }

    pub(crate) fn zeros(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            v: vec![0.; row * col],
        }
    }

    pub(crate) fn ones(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            v: vec![1.; row * col],
        }
    }

    pub(crate) fn new_row(row: &[f64]) -> Self {
        Self {
            row: 1,
            col: row.len(),
            v: row.to_vec(),
        }
    }

    pub(crate) fn hstack(&self, other: &Self) -> Self {
        assert_eq!(self.row, other.row);
        let row = self.row;
        let col = self.col + other.col;
        let mut v = vec![0.; self.row * (self.col + other.col)];
        for r in 0..self.row {
            for c in 0..self.col {
                v[r * col + c] = self.v[r * self.col + c];
            }
            for c in 0..other.col {
                v[r * col + c + self.col] = other.v[r * other.col + c];
            }
        }
        Self {
            row, col, v,
        }
    }

    pub(crate) fn eye(size: usize) -> Self {
        let mut v = vec![0.; size * size];
        for i in 0..size {
            v[i * size + i] = 1.;
        }
        Self {
            row: size,
            col: size,
            v,
        }
    }

    pub(crate) fn t(&self) -> Self {
        let mut v = vec![0.; self.row * self.col];
        for r in 0..self.row {
            for c in 0..self.col {
                v[c * self.row + r] = self.v[r * self.col + c];
            }
        }
        Self {
            row: self.col,
            col: self.row,
            v,
        }
    }

    pub(crate) fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            row: self.row,
            col: self.col,
            v: self.v.iter().copied().map(f).collect(),
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.v[r * self.col + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.v[r * self.col + c]
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for r in 0..self.row {
            if r != 0 {
                write!(f, "  ")?;
            }
            for c in 0..self.col {
                write!(f, "{}", self.v[r * self.col + c])?;
                if c != self.col - 1 {
                    write!(f, " ")?;
                }
            }
            if r != self.row - 1 {
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
        assert_eq!(self.row, other.row);
        assert_eq!(self.col, other.col);
        let mut v = self.v;
        for (v, o) in v.iter_mut().zip(other.v.iter()) {
            *v += o;
        }
        Self { row: self.row, col: self.col, v }
    }
}

#[test]
fn test_add() {
    let a = Matrix::eye(3);
    let b = Matrix::eye(3);
    let c = a + b;
    assert_eq!(c, Matrix::new([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]));
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.col, rhs.row);
        let mut v = vec![0.; self.row * rhs.col];
        for r in 0..self.row {
            for c in 0..rhs.col {
                v[r * rhs.col + c] = self.v[r * self.col..(r + 1) * self.col].iter().zip(0..self.col).map(|(s, oi)| {
                    s * rhs.v[oi * rhs.col + c]
                }).fold(0., |acc, cur| acc + cur)
            };
        }
        Matrix {
            row: self.row,
            col: rhs.col,
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