import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix as symMatrix
from tqdm import tqdm

from basis import Basis
from determinat import Determinat
from gramSchmidt import GramSchmidt
from inverse import Inverse
from matrix import FloatMatrix, Matrix
from matrixAdd import MatrixAdd
from matrixSubtract import MatrixSubtract
from multiply import Multiply
from rref import RREF
from scalarMultiply import ScalarMultiply
from unitVector import UnitVector
from vectorDot import VectorDot
from vectorLength import Length

# ============================
# Core Test Functions and Corresponding Random Test Functions
# ============================


# --- testMatrix ---
def testMatrix(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 1,
    min_: int = 0,
    max_: int = 100,
    verbose: bool = False,
):
    """
    Tests the creation of two matrices A and B with specified dimensions and value ranges.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices.

    Returns:
        None
    """
    if verbose:
        print(f"Making A and B\n")

    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")


def randomTestMatrix(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrix function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Always returns True as testMatrix does not perform validations.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = random.randint(0, maxSize)
    Bcols = random.randint(0, maxSize)

    try:
        testMatrix(
            Arows=Arows,
            Acols=Acols,
            Brows=Brows,
            Bcols=Bcols,
            min_=minSize,
            max_=maxSize,
            verbose=verbose,
        )
        return True
    except Exception as e:
        if verbose:
            print(f"Exception in randomTestMatrix: {e}")
        return False


# --- testVectorDot ---
def testVectorDot(
    Arows: int = 4,
    Acols: int = 1,
    Brows: int = 1,
    Bcols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the VectorDot operation between two vectors A and B.

    Args:
        Arows (int): Number of rows in vector A.
        Acols (int): Number of columns in vector A.
        Brows (int): Number of rows in vector B.
        Bcols (int): Number of columns in vector B.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vectors and results.

    Returns:
        bool: True if the dot product is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)
    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"A:\n{a}")
        print(f"B:\n{b}")
        print(f"Calculating Dot Product of A and B\n")

    try:
        c = VectorDot(a, b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d = np.dot(a_flat, b_flat)

    correct = c == d

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestVectorDot(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testVectorDot function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testVectorDot function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        Arows = vectorElemCount
        Acols = 1
    else:
        Arows = 1
        Acols = vectorElemCount

    if random.randrange(0, 2) == 0:
        Brows = vectorElemCount
        Bcols = 1
    else:
        Brows = 1
        Bcols = vectorElemCount

    testResult = testVectorDot(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMultiply ---
def testMultiply(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 2,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Multiply operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the multiplication is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")
        print(f"Multiplying A and B\n")

    try:
        c = Multiply(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"c:\n{c}")

    d = np.matmul(a.matrix, b.matrix)
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMultiply(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMultiply function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMultiply function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Acols  # Ensures multiplication is possible
    Bcols = random.randint(0, maxSize)

    testResult = testMultiply(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMatrixAdd ---
def testMatrixAdd(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 4,
    Bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the MatrixAdd operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the addition is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")

    try:
        c = MatrixAdd(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = a.matrix + b.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMatrixAdd(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrixAdd function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMatrixAdd function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures addition is possible
    Bcols = Acols

    testResult = testMatrixAdd(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMatrixSubtract ---
def testMatrixSubtract(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 4,
    Bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the MatrixSubtract operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the subtraction is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")

    try:
        c = MatrixSubtract(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = a.matrix - b.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMatrixSubtract(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrixSubtract function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMatrixSubtract function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures subtraction is possible
    Bcols = Acols

    testResult = testMatrixSubtract(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testVectorLength ---
def testVectorLength(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Length operation for a vector.

    Args:
        rows (int): Number of rows in the vector.
        cols (int): Number of columns in the vector.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vector and results.

    Returns:
        bool: True if the length is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:
        c = Length(vector=a)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        return False

    if verbose:
        print(f"c:\n{c}")

    d = np.linalg.norm(a.matrix)

    correct = c == d

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestVectorLength(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testVectorLength function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testVectorLength function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        rows = vectorElemCount
        cols = 1
    else:
        rows = 1
        cols = vectorElemCount

    testResult = testVectorLength(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testUnitVector ---
def testUnitVector(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the UnitVector operation for a vector.

    Args:
        rows (int): Number of rows in the vector.
        cols (int): Number of columns in the vector.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vector and results.

    Returns:
        bool: True if the unit vector is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:
        c = UnitVector(vector=a)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        return False

    if verbose:
        print(f"c:\n{c}")

    if np.linalg.norm(a.matrix) == 0:
        d = a.matrix
    else:
        d = a.matrix / np.linalg.norm(a.matrix)

    correct = c.shape == d.shape

    if correct:
        c_flat = c.flatten()
        d_flat = d.flatten()
        for row in range(c_flat.shape[0]):
            correct = abs(c_flat[row] - d_flat[row]) < 0.0001
            if not correct:
                print(c_flat[row][0])
                print(d_flat[row][0])
                break

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"a:\n{a}")
        print(f"ShapeA: {a.shape}")
        print(f"LengthA Own: {Length(vector=a)}")
        print(f"LengthA numpy: {np.linalg.norm(a.matrix)}\n")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestUnitVector(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testUnitVector function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testUnitVector function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        rows = vectorElemCount
        cols = 1
    else:
        rows = 1
        cols = vectorElemCount

    testResult = testUnitVector(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testScalarMultiply ---
def testScalarMultiply(
    scalar: int | float = 5,
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the ScalarMultiply operation on a matrix.

    Args:
        scalar (int | float): Scalar value to multiply with the matrix.
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the scalar multiplication is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:
        c = ScalarMultiply(scalar=scalar, matrix=a)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{scalar=}\n{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = scalar * a.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = abs(c[row][col] - d[row][col]) < 0.00001
            if not correct:
                print(c[row][col])
                print(d[row][col])
                break

    if not correct:
        print(f"{scalar=}\n{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"A:\n{a}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestScalarMultiply(
    minSize: int = -100,
    maxSize: int = 100,
    verbose: bool = False,
) -> bool:
    """
    Generates random dimensions and scalar values to test the testScalarMultiply function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and scalar values.
        maxSize (int): Maximum size for the matrix dimensions and scalar values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testScalarMultiply function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)

    scalar = random.randint(minSize, maxSize)

    testResult = testScalarMultiply(
        scalar=scalar,
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testRREF ---
def testRREF(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Reduced Row Echelon Form (RREF) operation on a matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the RREF is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    rrefA = RREF(matrix=a)

    sympyRREF = symMatrix(a.matrix)
    sympyRREF = sympyRREF.rref()[0]
    sympyRREF = FloatMatrix(np.array(sympyRREF).astype(float))

    if verbose:
        print(f"rrefA:\n{rrefA}")
        print(f"sympyRREF:\n{sympyRREF}")

    correct = rrefA.shape == sympyRREF.shape

    for row in range(rrefA.shape[0]):
        for col in range(rrefA.shape[1]):
            if not correct:
                break
            correct = abs(rrefA[row][col] - sympyRREF[row][col]) < 0.00001
            if not correct:
                print(rrefA[row][col])
                print(sympyRREF[row][col])
                break

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"rrefA:\n{rrefA}")
        print(f"sympyRREF:\n{sympyRREF}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestRREF(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testRREF function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testRREF function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)
    augmented = 0  # Unused parameter, kept for interface consistency

    testResult = testRREF(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testBasis ---
def testBasis(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Basis operation to find a basis for the column space of a matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the basis is correct, False otherwise.
    """

    def basisOfVectors(vectors):
        """
        Returns a set of linearly independent vectors forming the basis of the input matrix.

        Parameters:
        vectors (numpy.ndarray): A 2D array where each column represents a vector.

        Returns:
        numpy.ndarray: A matrix containing the linearly independent vectors as columns.
        """
        matrix = vectors.astype(float)
        n_rows, n_cols = matrix.shape
        pivot_columns = []
        row = 0

        for col in range(n_cols):
            if row >= n_rows:
                break
            if matrix[row, col] == 0:
                for r in range(row + 1, n_rows):
                    if matrix[r, col] != 0:
                        matrix[[row, r]] = matrix[[r, row]]
                        break
            if matrix[row, col] != 0:
                pivot_columns.append(col)
                matrix[row] = matrix[row] / matrix[row, col]
                for r in range(n_rows):
                    if r != row:
                        matrix[r] -= matrix[r, col] * matrix[row]
                row += 1

        independent_vectors = vectors[:, pivot_columns]

        return independent_vectors

    a = Matrix(rows, cols, min_, max_)

    a = Matrix(
        np.array(
            [
                [-6, -8, -68, -76, -52],
                [-63, -90, -90, -18, -75],
                [-9, -13, 2, -63, -59],
                [-90, -70, -5, -40, -47],
                [-27, -71, -92, -65, -20],
            ]
        )
    )

    if verbose:
        print(f"A:\n{a}")
    basisA = Basis(span=a)
    c = basisOfVectors(a.matrix)

    if verbose:

        print(f"basisA:\n{basisA}")
        print(f"C:\n{c}")

    correct = basisA.shape == c.shape

    if (
        basisA.shape[0] == 1
        and basisA.shape[1] == 1
        and c.shape[0] == 1
        and c.shape[1] == 1
    ):

        if c[0][0] == 0 and a[0][0] == 0:

            return True

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape basisA: {basisA.shape}")
        print(f"Shape C: {c.shape}\n")

    for row in range(basisA.shape[0]):
        for col in range(basisA.shape[1]):

            if not correct:

                break

            correct = abs(basisA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"basisA: {basisA[row][col]}")
                print(f"C: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"basisA:\n{basisA}\n")
        print(f"C:\n{c}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestBasis(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testBasis function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = testBasis(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testDeterminant ---
def testDeterminant(
    rows: int = 4,
    cols: int = 4,
    min_: int = -10,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Determinant operation on a square matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the determinant is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    detA = Determinat(matrix=a)
    b = np.linalg.det(a.matrix)

    if verbose:
        print(f"detA:\n{detA}")
        print(f"B:\n{b}")

    if abs(detA) > 100000000:
        correct = abs((detA - b) / b) < 0.000001
    else:
        correct = abs(detA - b) < 0.00001

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"A:\n{a}")
        print(f"detA:\n{detA}")
        print(f"B:\n{b}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestDeterminant(
    minSize: int = -100, maxSize: int = 12, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random square dimensions and values to test the testDeterminant function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testDeterminant function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = testDeterminant(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


# --- testInverse ---
def testInverse(
    rows: int = 4,
    cols: int = 4,
    min_: int = -10,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Inverse operation on a square matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the inverse is correct or None if the matrix is singular, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    invA = Inverse(matrix=a)

    if Determinat(a) == 0:
        return invA is None

    try:
        c = np.linalg.inv(a.matrix)
    except np.linalg.LinAlgError:
        if verbose:
            print("Matrix is singular and cannot be inverted.")
        return invA is None

    if verbose:
        print(f"invA:\n{invA}")
        print(f"C:\n{c}")

    correct = invA.shape == c.shape

    for row in range(invA.shape[0]):
        for col in range(invA.shape[1]):
            if not correct:
                break
            correct = abs(invA[row][col] - c[row][col]) < 0.00001
            if not correct:
                print(invA[row][col])
                print(c[row][col])
                break

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"invA:\n{invA}")
        print(f"C:\n{c}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestInverse(
    minSize: int = -100, maxSize: int = 100, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random square dimensions and values to test the testInverse function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testInverse function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = testInverse(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


# ============================
# Utility Functions
# ============================


def calculateTimeComplexity(
    func,
    minSize: int = -100,
    maxSize: int = 100,
    maxVal: int = 100,
    verbose: bool = True,
):
    """
    Measures the execution time of a given function for different input sizes.

    Args:
        func: The function to be tested, which accepts rows, cols, min_, and max_ as parameters.
        minSize (int): Minimum size for the input (default -100).
        maxSize (int): Maximum size for the input (default 100).
        maxVal (int): Maximum absolute value for the input values (default 100).
        verbose (bool): If True, prints execution details and plots results (default True).

    Returns:
        None
    """
    runs = []
    if verbose:
        print(f"Function: {func.__name__}")

    for size in range(maxSize):
        startTime = time.time()

        if (
            func.__name__ == "randomTestInverse"
            or func.__name__ == "randomTestDeterminant"
        ):
            testResult = func(minSize=-maxVal, maxSize=maxVal, verbose=False)
        else:
            testResult = func(minSize=minSize, maxSize=maxSize, verbose=False)

        endTime = time.time()

        if not testResult:
            print(f"{func.__name__}")
            break

        runs.append((size, endTime - startTime))

        if verbose:
            print(f"\nSize: {size}\n\tTime: {endTime - startTime} s")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sizes, times = zip(*runs) if runs else ([], [])
    ax.plot(sizes, times)
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Execution Time (s)")

    plt.savefig(f"{func.__name__}Complexity.png")

    if verbose and runs:
        plt.show()


# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":

    minSize = -100
    maxSize = 100

    noErrors = True

    # Test Basis
    if noErrors:
        for _ in tqdm(range(20000), desc="testBasis"):
            testResult = randomTestBasis(minSize=minSize, maxSize=5, verbose=False)

            if not testResult:

                print("Basis")
                noErrors = False

                break

    # Test Inverse
    if noErrors:
        for _ in tqdm(range(100), desc="testInverse"):
            testResult = randomTestInverse(
                minSize=minSize, maxSize=9, maxVal=maxSize, verbose=False
            )
            if not testResult:
                print("Inverse")
                noErrors = False
                break

    # Test Determinant
    if noErrors:
        for _ in tqdm(range(100), desc="testDeterminant"):
            testResult = randomTestDeterminant(
                minSize=minSize, maxSize=8, maxVal=maxSize, verbose=False
            )
            if not testResult:
                print("Determinant")
                noErrors = False
                break

    # Test VectorDot
    if noErrors:
        for _ in tqdm(range(1000), desc="testVectorDot"):
            testResult = randomTestVectorDot(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("VectorDot")
                noErrors = False
                break

    # Test Multiply
    if noErrors:
        for _ in tqdm(range(10), desc="testMultiply"):
            testResult = randomTestMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("Multiply")
                noErrors = False
                break

    # Test MatrixAdd
    if noErrors:
        for _ in tqdm(range(100), desc="testMatrixAdd"):
            testResult = randomTestMatrixAdd(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("MatrixAdd")
                noErrors = False
                break

    # Test MatrixSubtract
    if noErrors:
        for _ in tqdm(range(100), desc="testMatrixSubtract"):
            testResult = randomTestMatrixSubtract(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("MatrixSubtract")
                noErrors = False
                break

    # Test ScalarMultiply
    if noErrors:
        for _ in tqdm(range(100), desc="testScalarMultiply"):
            testResult = randomTestScalarMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("ScalarMultiply")
                noErrors = False
                break

    # Test VectorLength
    if noErrors:
        for _ in tqdm(range(1000), desc="testVectorLength"):
            testResult = randomTestVectorLength(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("VectorLength")
                noErrors = False
                break

    # Test UnitVector
    if noErrors:
        for _ in tqdm(range(10000), desc="testUnitVector"):
            testResult = randomTestUnitVector(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("UnitVector")
                noErrors = False
                break

    # Test RREF
    if noErrors:
        for _ in tqdm(range(250), desc="testRREF"):
            testResult = randomTestRREF(minSize=minSize, maxSize=maxSize, verbose=False)
            if not testResult:
                print("RREF")
                noErrors = False
                break
