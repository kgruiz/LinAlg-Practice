import numpy as np
import sympy
from sympy import Basic, Expr, Poly, Symbol, eye, pprint, pretty, simplify
from sympy.abc import lamda

from Base2Multiply import *
from Base2RREF import *
from Base2VectorDot import *
from Basis import *
from Determinat import *
from GramSchmidt import *
from Inverse import *
from Matrix import *
from MatrixAdd import *
from MatrixMultiply import *
from MatrixSubtract import *
from QRDecomposition import *
from RREF import *
from ScalarMultiply import *
from Transpose import *
from UnitVector import *
from VectorDot import *
from VectorLength import *


def Bold(text: str) -> str:

    return f"\033[1m{text}\033[0m"


class Eigenvalue:
    """
    Represents an eigenvalue with its value, multiplicity, and identifier.

    Attributes
    ----------
    value : Union[int, float, Basic]
        The numerical value of the eigenvalue.
    multiplicity : int
        The number of times the eigenvalue appears.
    index : Union[int, str, Symbol], optional
        An identifier for the eigenvalue.

    Methods
    -------
    __init__(value, multiplicity, index=None)
        Initializes the Eigenvalue instance.
    __str__()
        Returns a string representation of the Eigenvalue.
    __repr__()
        Returns an unambiguous string representation of the Eigenvalue.
    """

    def __init__(
        self,
        value: Union[int, float, Basic],
        multiplicity: int,
        index: Union[int, str, Symbol] = None,
    ):
        """
        Initialize an Eigenvalue instance.

        Parameters
        ----------
        value : Union[int, float, Basic]
            The numerical value of the eigenvalue.
        multiplicity : int
            The multiplicity indicating how many times the eigenvalue occurs.
        index : Union[int, str, Symbol], optional
            An optional identifier for the eigenvalue.
        """
        self.value = value
        self.multiplicity = multiplicity
        self.index = index

    def __str__(self) -> str:
        """
        Return a readable string representation of the eigenvalue.

        Returns
        -------
        str
            A formatted string displaying the eigenvalue and its multiplicity.
        """
        valueStr = pretty(self.value)

        if self.index is not None:

            lambdaStr = pretty(Symbol(f"{lamda}_{self.index}"))
        else:

            lambdaStr = pretty(Symbol(f"{lamda}"))

        return (
            f"{lambdaStr}: {valueStr} ({pretty(Symbol('m_a'))} = {self.multiplicity})"
        )

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the eigenvalue.

        Returns
        -------
        str
            A string that can be used to recreate the Eigenvalue instance.
        """
        return self.__str__()


def GetEigenvalues(A: Union[FloatMatrix, Matrix], verbose: bool = False) -> tuple:
    """
    Calculate the real and complex eigenvalues of a square matrix.

    Parameters
    ----------
    A : Union[FloatMatrix, Matrix]
        The input square matrix for which to compute eigenvalues.
    verbose : bool, optional
        If True, prints detailed computation steps. Default is False.

    Returns
    -------
    tuple
        A tuple containing two sets:
        - Set of real eigenvalues.
        - Set of complex eigenvalues.

    Raises
    ------
    Exception
        If the input matrix is not square.
    """
    if A.numRows != A.numCols:

        raise Exception("Matrix must be square")

    if verbose:

        print("Input Matrix A:")
        print(A)
        print()

    idnMatrix = Idn(A.numRows)

    if verbose:

        print(f"Identity Matrix ({pretty(Symbol(f'I_{A.numRows}'))}):")
        print(idnMatrix)
        print()

    lambdaIdn = ScalarMultiply(scalar=lamda, matrix=idnMatrix)

    if verbose:

        print(f"{pretty(lamda * Symbol(f'I_{A.numRows}'))}:")
        print(lambdaIdn)
        print()

    characteristicMatrix = MatrixSubtract(matrixA=A, matrixB=lambdaIdn)

    if verbose:

        print(
            f"Characteristic Matrix ({pretty(Symbol('A') - lamda * Symbol(f'I_{A.numRows}'))}):"
        )
        pprint(characteristicMatrix)
        print()

    characteristicEquation = simplify(Determinat(matrix=characteristicMatrix))

    if verbose:

        print("Characteristic Equation:")
        pprint(characteristicEquation)
        print()

    if verbose:

        print("Factored Characteristic Equation:")
        pprint(sympy.factor(characteristicEquation))
        print()

    characteristicPoly = Poly(characteristicEquation)

    eigenValuesRaw = characteristicPoly.all_roots()

    realEigenvaluesRaw = characteristicPoly.real_roots()

    realEigenvalues = set()
    complexEigenvalues = set()

    for eigenValueRaw in eigenValuesRaw:

        if eigenValueRaw not in realEigenvaluesRaw:

            complexEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            complexEigenvalues.add(complexEigenvalue)

        else:

            realEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            realEigenvalues.add(realEigenvalue)

    realEigenvalues = set(sorted(realEigenvalues, key=lambda x: x.value))
    complexEigenvalues = set(sorted(complexEigenvalues, key=lambda x: x.value))

    for i, realEigenvalue in enumerate(realEigenvalues):

        realEigenvalue.index = i

    for i, complexEigenvalue in enumerate(complexEigenvalues):

        complexEigenvalue.index = f"{i}i"

    if verbose:

        print("Real Eigenvalues:")

        for eigenValue in realEigenvalues:

            print(f"  {eigenValue}")

    if verbose and len(complexEigenvalues) > 0:

        print("Complex Eigenvalues:")

        for complexEigenvalue in complexEigenvalues:

            print(f"  {complexEigenvalue}")

        print()

    return realEigenvalues, complexEigenvalues


def GetEigenvectors(
    A: Union[FloatMatrix, Matrix], makeUnit: bool = True, verbose: bool = False
) -> None:
    """
    Compute and display the eigenvectors associated with each eigenvalue of a matrix.

    Parameters
    ----------
    A : Union[FloatMatrix, Matrix]
        The input square matrix for which to compute eigenvectors.
    makeUnit : bool, optional
        If True, normalizes the eigenvectors to unit length. Default is True.
    verbose : bool, optional
        If True, prints detailed computation steps. Default is False.

    Returns
    -------
    None
        This function prints the eigenvectors directly.

    Raises
    ------
    Exception
        If eigenvalues are not calculated before computing eigenvectors.
    """
    if verbose:

        print("\nComputing Eigenvalues")
        print("-" * 75)

    realEigenvalues, complexEigenvalues = GetEigenvalues(A=A, verbose=verbose)

    print()

    if len(complexEigenvalues) > 0:

        eigenValues = set(realEigenvalues).update(set(complexEigenvalues))

    else:

        eigenValues = set(realEigenvalues)

    idnMatrix = Idn(A.numRows)

    if verbose:

        print("\nComputing Eigenvectors")
        print("-" * 75)

    if verbose:

        print("Input Matrix A:")
        print(A)
        print()

    if verbose:

        print(f"Identity Matrix ({pretty(Symbol(f'I_{A.numRows}'))}):")
        print(idnMatrix)
        print()

    for eigenValue in eigenValues:

        if verbose:

            print(f"\n\nEigenvalue: {eigenValue}")
            print("-" * 50)

        if not isinstance(eigenValue, Eigenvalue):

            raise Exception("Eigenvalues must be calculated first")

        eigenIdn = ScalarMultiply(scalar=eigenValue.value, matrix=idnMatrix)

        if verbose:

            pass

        eigenvectorEquationMatrix = MatrixSubtract(matrixA=A, matrixB=eigenIdn)

        if verbose:

            print(
                f"Eigenvector Equation Matrix ({pretty(Symbol('A') - lamda * Symbol(f'I_{A.numRows}'))}):"
            )
            pprint(eigenvectorEquationMatrix)
            print()

        zeroVector = Matrix(np.zeros((A.numRows, 1)))

        augmentedEigenvectorEquation = MatrixAppend(
            matrixA=eigenvectorEquationMatrix, matrixB=zeroVector
        )

        augmentedEigenvectorEquation.SetNumAugmented(1)

        if verbose:

            print(
                f"Augmented Eigenvector Matrix (({pretty(Symbol('A') - lamda * Symbol(f'I_{A.numRows}'))}){Bold(text="v")} = {Bold(text="0")}):"
            )
            pprint(augmentedEigenvectorEquation)
            print()

        rowReducedEigenvectorMatrix = RREF(
            matrix=augmentedEigenvectorEquation, augmentedColCount=1
        )

        if verbose:

            print(f"RREF of Augmented Eigenvector Matrix:")
            pprint(rowReducedEigenvectorMatrix)
            print()

        pivotColIndices = GetPivotColumns(
            matrix=rowReducedEigenvectorMatrix, numAugmented=1
        )
        freeVarIndices = GetFreeVariables(
            matrix=rowReducedEigenvectorMatrix, numAugmented=1
        )

        if verbose:

            pivotCols = "".join([f"{colIndex}, " for colIndex in pivotColIndices[:-1]])

            if len(pivotColIndices) > 0:

                pivotCols += f"{pivotColIndices[-1]}"

            freeVars = "".join([f"{colIndex}, " for colIndex in freeVarIndices][:-1])

            if len(freeVarIndices) > 0:

                freeVars += f"{freeVarIndices[-1]}"

            print(f"Pivot Columns ({len(pivotColIndices)}): {pivotCols}")
            print(f"Free Variables ({len(freeVarIndices)}): {freeVars}")


A = FloatMatrix(np.array([[1, 2], [4, 3]]))


eigenVectors = GetEigenvectors(A=A, verbose=True)
