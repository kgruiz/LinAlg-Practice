from collections import defaultdict
from typing import List

import numpy as np
import sympy
from ordered_set import OrderedSet
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
            f"{lambdaStr}: {valueStr} ({pretty(Symbol("m_a"))} = {self.multiplicity})"
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

        return f"{self.eigenvalue}:\n({pretty(Symbol(f"m(g)"))} = {self.geomMultiplicity})\n{self.vector}"

    def __repr__(self) -> str:

        return self.__str__()


def GetVectorColNum(variable: Symbol) -> int:

    colNum = int(variable.name.split("_")[1])

    return colNum


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

        print("\nComputing Eigenvalues")
        print("-" * 75)
        print()

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

    realEigenvalues = OrderedSet()
    complexEigenvalues = OrderedSet()

    addedRawRealEigenvalues = set()
    addedRawComplexEigenvalues = set()

    for eigenValueRaw in eigenValuesRaw:

        if eigenValueRaw not in realEigenvaluesRaw:

            eigenValueRaw = eigenValueRaw.round(2)

            complexEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            if eigenValueRaw not in addedRawComplexEigenvalues:

                complexEigenvalues.add(complexEigenvalue)
                addedRawComplexEigenvalues.add(eigenValueRaw)

        else:

            eigenValueRaw = eigenValueRaw.round(2)

            realEigenvalue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            if eigenValueRaw not in addedRawRealEigenvalues:

                realEigenvalues.add(realEigenvalue)
                addedRawRealEigenvalues.add(eigenValueRaw)

    realEigenvalues = OrderedSet(sorted(realEigenvalues, key=lambda x: x.value))

    if verbose:

        for a in complexEigenvalues:

            print(f"a: {a}")

        print()

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

        print()

    if verbose and len(complexEigenvalues) > 0:

        print("Complex Eigenvalues:")

        for complexEigenvalue in complexEigenvalues:

            print(f"  {complexEigenvalue}")

        print()

        if verbose:

            print("=" * 75)
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

    realEigenvalues, complexEigenvalues = GetEigenvalues(A=A, verbose=verbose)

    if len(complexEigenvalues) > 0:

        # TODO: handle complex eigenvalues
        eigenValues = OrderedSet(realEigenvalues).union(OrderedSet(complexEigenvalues))

    else:

        eigenValues = OrderedSet(realEigenvalues)

    eigenValues = OrderedSet(realEigenvalues)

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

    allEigenvectors = []

    for eigenValue in eigenValues:

        if verbose:

            print(f"\n\nEigenvalue: {eigenValue}")
            print("-" * 50)

        if not isinstance(eigenValue, Eigenvalue):

            raise Exception("Eigenvalues must be calculated first")

        eigenIdn = ScalarMultiply(scalar=eigenValue.value, matrix=idnMatrix)

        if verbose:

            print(
                f"{pretty(Symbol(f"{lamda}_{eigenValue.index}") * Symbol(f'I_{A.numRows}'))}:"
            )
            print(eigenIdn)
            print()

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

        pivotVars = []
        freeVars = []

        for pivotVarCol in pivotColIndices:

            pivotVars.append(Symbol(f"v_{pivotVarCol}"))

        for freeVarCol in freeVarIndices:

            freeVars.append(Symbol(f"v_{freeVarCol}"))

        if verbose:

            pivotColsIndicesStr = "".join(
                [f"{colIndex}, " for colIndex in pivotColIndices[:-1]]
            )
            pivotVarsStr = "".join(
                [f"{pretty(pivotVar)}, " for pivotVar in pivotVars[:-1]]
            )

            if len(pivotColIndices) > 0:

                pivotColsIndicesStr += f"{pivotColIndices[-1]}"
                pivotVarsStr += f"{pretty(pivotVars[-1])}"

            freeVarsIndicesStr = "".join(
                [f"{colIndex}, " for colIndex in freeVarIndices][:-1]
            )
            freeVarsStr = "".join([f"{pretty(freeVar)}, " for freeVar in freeVars[:-1]])

            if len(freeVarIndices) > 0:

                freeVarsIndicesStr += f"{freeVarIndices[-1]}"
                freeVarsStr += f"{pretty(freeVars[-1])}"

            if len(pivotColIndices) == 1:

                print(f"Pivot Column ({len(pivotColIndices)}): {pivotColsIndicesStr}")
                print(f"Pivot Variable ({len(pivotColIndices)}): {pivotVarsStr}\n")

            else:

                print(f"Pivot Columns ({len(pivotColIndices)}): {pivotColsIndicesStr}")
                print(f"Pivot Variables ({len(pivotColIndices)}): {pivotVarsStr}\n")

            if len(freeVarIndices) == 1:

                print(f"Free Column ({len(freeVarIndices)}): {freeVarsIndicesStr}")
                print(f"Free Variable ({len(freeVarIndices)}): {freeVarsStr}")

            else:

                print(f"Free Columns ({len(freeVarIndices)}): {freeVarsIndicesStr}")
                print(f"Free Variables ({len(freeVarIndices)}): {freeVarsStr}")

            print()

        vectorElementEquations: List[Eq] = []

        for colIndex in range(rowReducedEigenvectorMatrix.numCols - 1):

            if colIndex in freeVarIndices:

                vectorElementEquations.append(
                    Eq(Symbol(f"v_{colIndex}"), Symbol(f"v_{colIndex}"), evaluate=False)
                )

            elif colIndex in pivotColIndices:

                firstNonZeroRow = None

                for rowNum, elem in enumerate(rowReducedEigenvectorMatrix[:, colIndex]):

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

                    vectorElementEquations.append(
                        Eq(Symbol(f"v_{colIndex}"), 0, evaluate=False)
                    )

                elif len(rowValues) == 2:

                    rowExpr = Add(
                        rowValues[0][1] * Symbol(f"v_{rowValues[0][0]}"),
                        rowValues[1][1] * Symbol(f"v_{rowValues[1][0]}"),
                    )

                    rowEq = Eq(rowExpr, 0)

                    vectorElementEquations.append(rowEq)

                else:

                    rowExpr = Add(
                        rowValues[0][1] * Symbol(f"v_{rowValues[0][0]}"),
                        rowValues[1][1] * Symbol(f"v_{rowValues[1][0]}"),
                    )

                    for rowValue in rowValues[2:]:

                        rowExpr += rowValue[1] * Symbol(f"v_{rowValues[1][0]}")

                    rowEq = Eq(rowExpr, 0)

                    vectorElementEquations.append(rowEq)

            else:

                raise Exception(f"Invalid column index: {colIndex}")

        if verbose:

            print(f"Row Reducing Gives Equations:")

            for equation in vectorElementEquations:

                print(f"{pretty(equation)}")

            print()

        for eqNum, equation in enumerate(vectorElementEquations):

            equation: Eq

            terms: defaultdict = equation.lhs.as_coefficients_dict()

            if len(terms) == 1:

                variable = list(terms.keys())[0]
                coefficient = list(terms.values())[0]

                if coefficient != 1:

                    raise Exception(
                        f"Invalid pivot equation: {pretty(equation)}. Coefficient is not one."
                    )

                continue

            if equation.rhs != 0:

                raise Exception(
                    f"Invalid equation: {pretty(equation)}. Should equal zero."
                )

            eqPivotVars = []
            eqFreeVars = []

            for variable, coefficient in terms.items():

                if variable in pivotVars:

                    eqPivotVars.append((coefficient, variable))

                elif variable in freeVars:

                    eqFreeVars.append((coefficient, variable))

                else:

                    raise Exception(f"Invalid variable: {pretty(variable)}")

            if len(eqPivotVars) > 1:

                raise Exception(
                    f"Unexpectedly more than 1 pivot variable. Pivot variables:\n{pretty(eqPivotVars)}"
                )

            if len(eqPivotVars) == 0:

                raise Exception(
                    f"No pivot variables found in equation: {pretty(equation)}"
                )

            if eqPivotVars[0][0] < 0:

                isolatedEquationLHS = -eqPivotVars[0][0] * eqPivotVars[0][1]

            else:

                isolatedEquationLHS = eqPivotVars[0][0] * eqPivotVars[0][1]

            isolatedEquationRHS = Add(0, 0)

            for coefficient, variable in eqFreeVars:

                if coefficient < 0:

                    isolatedEquationRHS -= coefficient * variable

                else:

                    isolatedEquationRHS += coefficient * variable

            isolatedEquation = Eq(
                isolatedEquationLHS, isolatedEquationRHS, evaluate=False
            )

            vectorElementEquations[eqNum] = isolatedEquation

        if verbose:

            print(f"Isolated Free Variables:")

            for equation in vectorElementEquations:

                print(f"{pretty(equation)}")

            print()

        if len(freeVarIndices) == 1:

            eigenVector = np.empty(shape=(A.numCols,))

            for colNum in range(A.numCols):

                eigenVector[colNum] = -100

                freeVar = freeVars[0]

            colNum = GetVectorColNum(variable=freeVar)

            eigenVector[colNum] = 1

            for equation in vectorElementEquations:

                leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                if len(leftTerms) > 1:

                    raise Exception(
                        f"Invalid equation (more than one left variable): {pretty(equation)}"
                    )

                leftVariable, leftCoefficient = list(leftTerms.items())[0]

                if len(rightTerms) == 1:

                    rightVariable, rightCoefficient = list(rightTerms.items())[0]

                    if rightVariable == 1:

                        if rightCoefficient != 0:

                            raise Exception(
                                f"Invalid non-zero right hand side constant: {coefficient}"
                            )

                        colNum = GetVectorColNum(variable=leftVariable)

                        eigenVector[colNum] = 0

            for equation in vectorElementEquations:

                leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                leftVariable, leftCoefficient = list(leftTerms.items())[0]

                if len(rightTerms) == 1:

                    rightVariable, rightCoefficient = list(rightTerms.items())[0]

                    if rightVariable == 1:

                        continue

                    rightVarCol = GetVectorColNum(variable=rightVariable)

                    rightVarValue = eigenVector[rightVarCol]

                    leftColNum = GetVectorColNum(variable=leftVariable)

                    if leftCoefficient != 1:

                        raise Exception(
                            f"Invalid non-one left coefficient: {leftCoefficient}"
                        )

                    eigenVector[leftColNum] = rightCoefficient * rightVarValue

            for equation in vectorElementEquations:

                leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                leftVariable, leftCoefficient = list(leftTerms.items())[0]

                if len(rightTerms) > 1:

                    rightVal = 0

                    for variable, coefficient in rightTerms.items():

                        rightVarCol = GetVectorColNum(variable=rightVariable)

                        rightVarValue = eigenVector[rightVarCol]

                        rightVal += rightCoefficient * rightVarValue

                    leftColNum = GetVectorColNum(variable=leftVariable)

                    if leftCoefficient != 1:

                        raise Exception(
                            f"Invalid non-one left coefficient: {leftCoefficient}"
                        )

                    eigenVector[leftColNum] = rightVal

            allEigenvectors.append(
                Eigenvector(
                    eigenvalue=eigenValue,
                    vector=FloatMatrix(eigenVector),
                    geomMultiplicity=1,
                    algMultiplicity=eigenValue.multiplicity,
                )
            )

            if verbose:

                print(f"Found Eigenvector:")
                print(f"{"-"*25}")
                print(allEigenvectors[-1])
                print()

                print(f"{"="*50}")

        else:

            for freeVar in freeVars:

                eigenVector = np.empty(shape=(A.numCols,))

                for colNum in range(A.numCols):

                    eigenVector[colNum] = -100

                colNum = GetVectorColNum(variable=freeVar)

                eigenVector[colNum] = 1

                for otherFreeVars in freeVars:

                    if otherFreeVars != freeVar:

                        colNum = GetVectorColNum(variable=otherFreeVars)

                        eigenVector[colNum] = 0

                for equation in vectorElementEquations:

                    leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                    rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                    if len(leftTerms) > 1:

                        raise Exception(
                            f"Invalid equation (more than one left variable): {pretty(equation)}"
                        )

                    leftVariable, leftCoefficient = list(leftTerms.items())[0]

                    if len(rightTerms) == 1:

                        rightVariable, rightCoefficient = list(rightTerms.items())[0]

                        if rightVariable == 1:

                            if rightCoefficient != 0:

                                raise Exception(
                                    f"Invalid non-zero right hand side constant: {coefficient}"
                                )

                            colNum = GetVectorColNum(variable=leftVariable)

                            eigenVector[colNum] = 0

                for equation in vectorElementEquations:

                    leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                    rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                    leftVariable, leftCoefficient = list(leftTerms.items())[0]

                    if len(rightTerms) == 1:

                        rightVariable, rightCoefficient = list(rightTerms.items())[0]

                        if rightVariable == 1:

                            continue

                        rightVarCol = GetVectorColNum(variable=rightVariable)

                        rightVarValue = eigenVector[rightVarCol]

                        leftColNum = GetVectorColNum(variable=leftVariable)

                        if leftCoefficient != 1:

                            raise Exception(
                                f"Invalid non-one left coefficient: {leftCoefficient}"
                            )

                        eigenVector[leftColNum] = rightCoefficient * rightVarValue

                for equation in vectorElementEquations:

                    leftTerms: defaultdict = equation.lhs.as_coefficients_dict()
                    rightTerms: defaultdict = equation.rhs.as_coefficients_dict()

                    leftVariable, leftCoefficient = list(leftTerms.items())[0]

                    if len(rightTerms) > 1:

                        rightVal = 0

                        for variable, coefficient in rightTerms.items():

                            rightVarCol = GetVectorColNum(variable=rightVariable)

                            rightVarValue = eigenVector[rightVarCol]

                            rightVal += rightCoefficient * rightVarValue

                        leftColNum = GetVectorColNum(variable=leftVariable)

                        if leftCoefficient != 1:

                            raise Exception(
                                f"Invalid non-one left coefficient: {leftCoefficient}"
                            )

                        eigenVector[leftColNum] = rightVal

                allEigenvectors.append(
                    Eigenvector(
                        eigenvalue=eigenValue,
                        vector=FloatMatrix(eigenVector),
                        geomMultiplicity=1,
                        algMultiplicity=eigenValue.multiplicity,
                    )
                )

                if verbose:

                    print(f"Found Eigenvector:")
                    print(f"{"-"*25}")
                    print(allEigenvectors[-1])
                    print()

                    print(f"{"="*50}")

    if verbose:

        print(f"\nAll Eigenvectors:")
        print(f"{"-"*50}")

        for eigenvector in allEigenvectors:

            print(eigenvector)

            print()

        print(f"{"="*75}")

    return allEigenvectors


# A = FloatMatrix(np.array([[5, 1, 5, 1], [0, 4, 3, 1], [0, 2, 1, 4], [5, 5, 6, 1]]))
# A = FloatMatrix(np.array([[1, 2, 1], [0, 3, 0], [0, 0, 2]]))
A = FloatMatrix(np.array([[4, 0, 0], [0, 4, 0], [0, 0, 5]]))

eigenVectors = GetEigenvectors(A=A, verbose=True)
