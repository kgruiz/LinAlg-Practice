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

   Ensure you have `pip` installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   *If a `requirements.txt` file is not provided, install the necessary packages manually:*

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
A = Matrix(3, 3, min_=1, max_=10)
B = Matrix(3, 3, min_=1, max_=10)

print("Matrix A:")
print(A)

print("Matrix B:")
print(B)

# Add matrices
C = MatrixAdd(A, B)
print("A + B:")
print(C)

# Subtract matrices
D = MatrixSubtract(A, B)
print("A - B:")
print(D)

# Scalar multiplication
E = ScalarMultiply(5, A)
print("5 * A:")
print(E)

# Transpose
F = Transpose(A)
print("Transpose of A:")
print(F)

# Matrix multiplication
G = MatrixMultiply(A, B)
print("A * B:")
print(G)
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
rref_A = RREF(A)
print("RREF of A:")
print(rref_A)

# QR Decomposition
Q, R = QRDecomposition(A)
print("QR Decomposition of A:")
print("Q:")
print(Q)
print("R:")
print(R)

# Gram-Schmidt Orthogonalization
orthonormal_basis = GramSchmidt(A)
print("Orthonormal Basis from A:")
print(orthonormal_basis)

# Determinant
det_A = Determinat(A)
print("Determinant of A:")
print(det_A)

# Inverse (if A is invertible)
inv_A = Inverse(A)
if inv_A:
    print("Inverse of A:")
    print(inv_A)
else:
    print("Matrix A is singular and cannot be inverted.")

# Basis of Column Space
basis_A = Basis(A)
print("Basis of Column Space of A:")
print(basis_A)
```

### Vector Operations

```python
from VectorDot import VectorDot
from VectorLength import Length
from UnitVector import UnitVector

# Create two vectors
v1 = Matrix(1, 3, min_=1, max_=5)
v2 = Matrix(1, 3, min_=1, max_=5)

print("Vector v1:")
print(v1)

print("Vector v2:")
print(v2)

# Dot product
dot = VectorDot(v1, v2)
print("Dot Product of v1 and v2:")
print(dot)

# Vector length
length_v1 = Length(v1)
print("Length of v1:")
print(length_v1)

# Unit vector
unit_v1 = UnitVector(v1)
print("Unit Vector of v1:")
print(unit_v1)
```

### Utilities

```python
from MatrixAppend import MatrixAppend
from Transpose import Transpose
from Matrix import Matrix, FloatMatrix

# Create two matrices
A = Matrix(2, 3, min_=1, max_=5)
B = Matrix(2, 2, min_=1, max_=5)

print("Matrix A:")
print(A)

print("Matrix B:")
print(B)

# Append matrices horizontally
C = MatrixAppend(A, B, horizontalStack=True)
print("A appended with B horizontally:")
print(C)

# Append matrices vertically
D = MatrixAppend(A, B, horizontalStack=False)
print("A appended with B vertically:")
print(D)

# Transpose a matrix
transposed_A = Transpose(A)
print("Transposed A:")
print(transposed_A)
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
from Main import CalculateTimeComplexity, TestMultiply

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
from Main import CalculateTimeComplexity, TestMatrixMultiply

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