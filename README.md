# LinAlg-Practice

This repository contains various Python scripts and modules for performing linear algebra operations and testing their correctness. The project includes implementations for matrix operations, vector operations, and other linear algebra concepts.

## Project Structure

- `Main.py`: Core test functions and corresponding random test functions for various linear algebra operations.
- `Basis.py`: Functions related to finding the basis of a matrix.
- `RREF.py`: Functions for computing the Reduced Row Echelon Form (RREF) of a matrix.
- `Base2RREF.py`: Functions for computing the RREF of matrices in base 2.
- `Matrix.py`: Definitions for `Matrix` and `FloatMatrix` classes.
- `temp.py`: Utility script for listing files in the directory and performing regex-based modifications.

## Core Test Functions

The core test functions are defined in `Main.py` and include:

- `TestMatrix`
- `TestVectorDot`
- `TestMultiply`
- `TestMatrixAdd`
- `TestMatrixSubtract`
- `TestVectorLength`
- `TestTranspose`
- `TestUnitVector`
- `TestScalarMultiply`
- `TestRREF`
- `TestBase2RREF`
- `TestBasis`
- `TestQRDecomposition`
- `TestGramSchmidt`
- `TestGramSchmidtRandomSpan`
- `TestDeterminant`
- `TestInverse`

Each test function has a corresponding random test function that generates random inputs to test the core function.

## Running the Tests

To run the tests, execute the `Main.py` script. The script will run all the test functions and print the results. The tests are organized in a sequence, and the script will stop if any test fails.

```bash
python Main.py
```

The image describes the following content:

---

### Utility Functions
The project includes utility functions for measuring the time complexity of the test functions. These functions are defined in the `Main.py` script.

---

### Dependencies
The project requires the following Python packages:
- **numpy**
- **sympy**
- **tqdm**
- **matplotlib**

You can install these dependencies using pip:
```bash
pip install numpy sympy tqdm matplotlib
```