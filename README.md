# LinAlg-Practice

A comprehensive Python library designed to deepen my understanding of linear algebra through hands-on implementation. LinAlg-Practice offers advanced matrix operations, including Reduced Row Echelon Form (RREF), QR Decomposition, Gram-Schmidt Orthogonalization, matrix inversion, determinant calculation, and more. The accompanying tests in `Main.py` compare my implementations with established packages like NumPy to ensure accuracy and reliability.

## Table of Contents

- [Features](#features)
  - [Matrix Types](#matrix-types)
  - [Basic Matrix Operations](#basic-matrix-operations)
  - [Advanced Matrix Operations](#advanced-matrix-operations)
  - [Vector Operations](#vector-operations)
  - [Utilities](#utilities)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Operations](#basic-operations)
  - [Advanced Operations](#advanced-operations)
  - [Vector Operations](#vector-operations)
  - [Utilities](#utilities)
- [Testing Framework](#testing-framework)
- [Performance Analysis](#performance-analysis)

## Features

### Matrix Types

- **Matrix**: Represents an integer matrix with support for symbolic entries using `sympy`.
- **FloatMatrix**: Represents a floating-point matrix, allowing for decimal precision.
- **Base2Matrix**: Represents a binary matrix (elements are either 0 or 1), useful for applications in computer science and information theory.

### Basic Matrix Operations

- **Addition (`MatrixAdd`)**: Add two matrices of identical dimensions.
- **Subtraction (`MatrixSubtract`)**: Subtract one matrix from another of identical dimensions.
- **Scalar Multiplication (`ScalarMultiply`)**: Multiply a matrix by a scalar value.
- **Transpose (`Transpose`)**: Transpose a matrix, swapping rows with columns.

### Advanced Matrix Operations

- **Matrix Multiplication (`MatrixMultiply`)**: Multiply two matrices, ensuring the number of columns in the first matrix matches the number of rows in the second.
- **Reduced Row Echelon Form (`RREF` and `Base2RREF`)**: Convert a matrix to its reduced row echelon form, facilitating solutions to linear systems.
- **QR Decomposition (`QRDecomposition`)**: Decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R.
- **Gram-Schmidt Orthogonalization (`GramSchmidt`)**: Generate an orthonormal basis from a set of vectors.
- **Determinant Calculation (`Determinat`)**: Compute the determinant of a square matrix.
- **Matrix Inversion (`Inverse`)**: Calculate the inverse of an invertible square matrix.
- **Basis Calculation (`Basis`)**: Determine a basis for the column space of a matrix.
- **Matrix Appending (`MatrixAppend`)**: Append two matrices either horizontally or vertically.
- **Identity Matrix Creation (`Idn` and `FloatIdn`)**: Generate identity matrices of integer or floating-point types.
- **Ensuring Binary Constraints (`EnsureNoTwo`)**: Ensure that binary matrices do not contain invalid entries (e.g., the number 2).

### Vector Operations

- **Dot Product (`VectorDot`)**: Compute the dot product of two vectors.
- **Vector Length (`Length`)**: Calculate the Euclidean norm of a vector.
- **Unit Vector (`UnitVector`)**: Normalize a vector to have a unit length.

### Utilities

- **Matrix Reshaping and Flattening**: Convert between different matrix shapes as needed for various operations.
- **Symbolic Matrix Support**: Integrate with `sympy` to handle symbolic mathematics within matrices.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/LinAlg-Practice.git
   cd LinAlg-Practice
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install numpy sympy matplotlib tqdm
   ```

## Usage

### Basic Operations

```python
from Matrix import Matrix
from MatrixAdd import MatrixAdd
from MatrixSubtract import MatrixSubtract
from ScalarMultiply import ScalarMultiply
from Transpose import Transpose
from MatrixMultiply import MatrixMultiply

# Create two matrices
a = Matrix(3, 3, min_=1, max_=10)
b = Matrix(3, 3, min_=1, max_=10)

print(f"Matrix A:\n{a}")

print(f"Matrix B:\n{b}")

# Add matrices
c = MatrixAdd(a, b)
print(f"A + B:\n{c}")

# Subtract matrices
d = MatrixSubtract(a, b)
print(f"A - B:\n{d}")

# Scalar multiplication
e = ScalarMultiply(5, a)
print(f"5 * A:\n{e}")

# Transpose
f = Transpose(a)
print(f"Transpose of A:\n{f}")

# Matrix multiplication
g = MatrixMultiply(a, b)
print(f"A * B:\n{g}")
```

### Advanced Operations

```python
from RREF import RREF
from QRDecomposition import QRDecomposition
from GramSchmidt import GramSchmidt
from Determinat import Determinat
from Inverse import Inverse
from Basis import Basis

# Reduced Row Echelon Form
rrefA = RREF(a)
print(f"RREF of A:\n{rrefA}")

# QR Decomposition
q, r = QRDecomposition(a)
print(f"QR Decomposition of A:")
print(f"Q:\n{q}")
print(f"R:\n{r}")

# Gram-Schmidt Orthogonalization
orthonormalBasis = GramSchmidt(a)
print(f"Orthonormal Basis from A:\n{orthonormalBasis}")

# Determinant
detA = Determinat(a)
print(f"Determinant of A:\n{detA}")

# Inverse (if A is invertible)
invA = Inverse(a)
if invA:
    print(f"Inverse of A:\n{invA}")
else:
    print("Matrix A is singular and cannot be inverted.")

# Basis of Column Space
basisA = Basis(a)
print(f"Basis of Column Space of A:\n{basisA}")
```

### Vector Operations

```python
from VectorDot import VectorDot
from VectorLength import Length
from UnitVector import UnitVector

# Create two vectors
v1 = Matrix(1, 3, min_=1, max_=5)
v2 = Matrix(1, 3, min_=1, max_=5)

print(f"Vector v1:\n{v1}")

print(f"Vector v2:\n{v2}")

# Dot product
dotProduct = VectorDot(v1, v2)
print(f"Dot Product of v1 and v2:\n{dotProduct}")

# Vector length
lengthV1 = Length(v1)
print(f"Length of v1:\n{lengthV1}")

# Unit vector
unitV1 = UnitVector(v1)
print(f"Unit Vector of v1:\n{unitV1}")
```

### Utilities

```python
from MatrixAppend import MatrixAppend
from Transpose import Transpose
from Matrix import Matrix, FloatMatrix

# Create two matrices
a = Matrix(2, 3, min_=1, max_=5)
b = Matrix(2, 2, min_=1, max_=5)

print(f"Matrix A:\n{a}")

print(f"Matrix B:\n{b}")

# Append matrices horizontally
c = MatrixAppend(a, b, horizontalStack=True)
print(f"A appended with B horizontally:\n{c}")

# Append matrices vertically
d = MatrixAppend(a, b, horizontalStack=False)
print(f"A appended with B vertically:\n{d}")

# Transpose a matrix
transposedA = Transpose(a)
print(f"Transposed A:\n{transposedA}")
```

## Testing Framework

The `Main.py` script includes a suite of tests that validate the correctness of each matrix and vector operation. It performs both deterministic and randomized testing to ensure robustness. The tests compare the custom implementations with established packages like NumPy to ensure accuracy.

### Running Tests

To execute all tests, simply run the `Main.py` script:

```bash
python Main.py
```

This will sequentially run all the defined tests, providing progress feedback through progress bars and detailed output in case of failures.

### Test Descriptions

- **TestMatrix**: Validates matrix creation and basic properties.
- **TestVectorDot**: Ensures correct computation of vector dot products.
- **TestMultiply**: Tests matrix multiplication for correctness.
- **TestMatrixAdd** and **TestMatrixSubtract**: Verify addition and subtraction operations.
- **TestVectorLength**: Checks the calculation of vector norms.
- **TestTranspose**: Validates matrix transposition.
- **TestUnitVector**: Ensures correct normalization of vectors.
- **TestScalarMultiply**: Tests scalar multiplication accuracy.
- **TestRREF** and **TestBase2RREF**: Validate the computation of Reduced Row Echelon Forms.
- **TestBasis**: Ensures the correctness of basis computation.
- **TestQRDecomposition**: Validates QR Decomposition against NumPy's implementation.
- **TestGramSchmidt**: Ensures the Gram-Schmidt process produces orthonormal bases.
- **TestDeterminant**: Verifies determinant calculations against NumPy.
- **TestInverse**: Tests matrix inversion against NumPy's results.
- **TestMatrixWithSymbols**: Ensures operations work with symbolic matrices.

### Randomized Testing

Each core test has a corresponding `RandomTest*` function that performs randomized testing across a range of input sizes and values. This approach helps in uncovering edge cases and ensuring the library's robustness.

### Performance Measurement

The `CalculateTimeComplexity` utility measures the execution time of functions across different input sizes and plots the results. This is useful for understanding the scalability and efficiency of the implemented algorithms.

**Example Usage:**

```python
from Main import CalculateTimeComplexity, RandomTestMultiply

# Measure time complexity of Matrix Multiply
CalculateTimeComplexity(
    func=RandomTestMultiply,
    minSize=-100,
    maxSize=100,
    maxVal=100,
    verbose=True
)
```

This will generate a plot named `RandomTestMultiplyComplexity.png` showcasing how the execution time scales with input size.

## Performance Analysis

The library includes utility functions to analyze the time complexity of its operations. The `CalculateTimeComplexity` function measures the execution time of a given function over varying input sizes and generates a plot to visualize performance trends.

### Example Usage

```python
from Main import CalculateTimeComplexity, RandomTestMultiply

# Measure time complexity of Matrix Multiply
CalculateTimeComplexity(
    func=RandomTestMultiply,
    minSize=-100,
    maxSize=100,
    maxVal=100,
    verbose=True
)
```

This will generate a plot named `RandomTestMultiplyComplexity.png` showcasing how the execution time scales with input size.