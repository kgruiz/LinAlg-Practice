import numpy as np
import sympy
from sympy import (
    Add,
    Basic,
    ComplexRootOf,
    CRootOf,
    Eq,
    Expr,
    Poly,
    Symbol,
    eye,
    im,
    pprint,
    pretty,
    re,
    simplify,
)
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
        if isinstance(value, ComplexRootOf):

            self.value = value.evalf()
            self.real, self.imaginary = value.as_real_imag()
        else:

            self.value = value
            self.real = None
            self.imaginary = None

        self.multiplicity = multiplicity
        self.index = index
        self.complex: bool = self.real is not None

    def __str__(self) -> str:
        """
        Return a readable string representation of the eigenvalue.

        Returns
        -------
        str
            A formatted string displaying the eigenvalue and its multiplicity.
        """
        if self.complex:

            valueStr = pretty(self.value.round(3))

        else:

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


class Eigenvector:

    def __init__(
        self,
        eigenvalue: Eigenvalue,
        vector: Matrix | FloatMatrix,
        geomMultiplicity: int,
        algMultiplicity: int,
    ):

        self.eigenvalue = eigenvalue
        self.vector = vector
        self.geomMultiplicity = geomMultiplicity
        self.algMultiplicity = algMultiplicity

    def __str__(self) -> str:

        return f"{self.eigenvalue}: {self.vector}"

    def __repr__(self) -> str:

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

            eigenValueRaw = eigenValueRaw.round(2)

            complexEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            complexEigenvalues.add(complexEigenvalue)

        else:

            eigenValueRaw = eigenValueRaw.round(2)

            realEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            realEigenvalues.add(realEigenvalue)

    realEigenvalues = set(sorted(realEigenvalues, key=lambda x: x.value))

    for a in complexEigenvalues:

        print(f"a: {a}")

    complexEigenvalues = sorted(
        complexEigenvalues, key=lambda x: (re(x.value), im(x.value))
    )

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

        # TODO: handle complex eigenvalues
        eigenValues = set(realEigenvalues).union(set(complexEigenvalues))

    else:

        eigenValues = set(realEigenvalues)

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

    print(realEigenvalues)

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

            if len(pivotColIndices) == 1:

                print(f"Pivot Column ({len(pivotColIndices)}): {pivotCols}")

            else:

                print(f"Pivot Columns ({len(pivotColIndices)}): {pivotCols}")

            if len(freeVarIndices) == 1:

                print(f"Free Variable ({len(freeVarIndices)}): {freeVars}")

            else:

                print(f"Free Variables ({len(freeVarIndices)}): {freeVars}")

            geomMultiplicity = len(freeVarIndices)

            eigenVectorElems = []

            for colIndex in range(rowReducedEigenvectorMatrix.numCols - 1):

                if colIndex in freeVarIndices:

                    eigenVectorElems.append(
                        Eq(Symbol(f"v_{colIndex}"), 1, evaluate=False)
                    )

                elif colIndex in pivotColIndices:

                    firstNonZeroRow = None

                    for rowNum, elem in enumerate(
                        rowReducedEigenvectorMatrix[:, colIndex]
                    ):

                        if elem != 0:

                            if firstNonZeroRow is not None:

                                raise Exception(
                                    f"Invalid pivot column: {colIndex}. More than one non-zero element in column."
                                )

                            firstNonZeroRow = rowNum

                    if firstNonZeroRow is None:

                        raise Exception(
                            f"Invalid pivot column: {colIndex}. No non-zero elements in column."
                        )

                    rowValues = []

                    for colNum, elem in enumerate(
                        rowReducedEigenvectorMatrix[firstNonZeroRow, :]
                    ):

                        if elem != 0:

                            if len(rowValues) > 0 and colNum < colIndex:

                                raise Exception(
                                    f"Invalid pivot column: {colIndex}. Not first non-zero element in row."
                                )

                            rowValues.append((colNum, elem))

                    if len(rowValues) == 0:

                        raise Exception(
                            f"Invalid pivot column: {colIndex}. No non-zero elements in row."
                        )

                    elif len(rowValues) == 1:

                        eigenVectorElems.append(
                            Eq(Symbol(f"v_{colIndex}"), 0, evaluate=False)
                        )

                    elif len(rowValues) == 2:

                        rowEq = Add(
                            rowValues[0][1] * Symbol(f"v_{rowValues[0][0]}"),
                            rowValues[1][1] * Symbol(f"v_{rowValues[1][0]}"),
                        )

                        print(pretty(rowEq))

                    else:

                        rowEq = Add(
                            rowValues[0][1] * Symbol(f"v_{rowValues[0][0]}"),
                            rowValues[1][1] * Symbol(f"v_{rowValues[1][0]}"),
                        )

                        for rowValue in rowValues[2:]:

                            rowEq += rowValue[1] * Symbol(f"v_{rowValues[1][0]}")

                        print(pretty(rowEq))

                else:

                    raise Exception(f"Invalid column index: {colIndex}")

            for elem in eigenVectorElems:

                print(f"{pretty(elem)}")

            raise SystemExit


# A = FloatMatrix(np.array([[1, 2], [4, 3]]))
A = FloatMatrix(np.array([[5, 4, 2, 1], [0, 1, 3, 5], [0, 0, 1, 4], [5, 2, 6, 1]]))


eigenVectors = GetEigenvectors(A=A, verbose=True)
