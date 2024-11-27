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
from MatrixSubtract import *
from Multiply import *
from QRDecomposition import *
from RREF import *
from ScalarMultiply import *
from Transpose import *
from UnitVector import *
from VectorDot import *
from VectorLength import *


class Eigenvalue:

    def __init__(
        self,
        value: int | float | Basic,
        multiplicity: int,
        index: int | str | Symbol = None,
    ):

        self.value = value
        self.multiplicity = multiplicity
        self.index = index

    def __str__(self):

        valueStr = pretty(self.value)

        if self.index is not None:

            lambdaStr = pretty(Symbol(f"{lamda}_{self.index}"))

        else:

            lambdaStr = pretty(Symbol(f"{lamda}"))

        return (
            f"{lambdaStr}: {valueStr} ({pretty(Symbol("m_a"))} = {self.multiplicity})"
        )

    def __repr__(self):

        return self.__str__()


def GetEigenvalues(A: Union[FloatMatrix, Matrix], verbose: bool = False):

    if A.numRows != A.numCols:

        raise Exception("Matrix must be square")

    if verbose:

        print("Input Matrix A:")
        print(A)
        print()

    idnMatrix = Idn(A.numRows)

    if verbose:

        print(f"Identity Matrix ({pretty(Symbol(f"I_{A.numRows}"))}):")
        print(idnMatrix)
        print()

    lambdaIdn = ScalarMultiply(scalar=lamda, matrix=idnMatrix)

    if verbose:

        print(f"{pretty(lamda * Symbol(f"I_{A.numRows}"))}:")
        print(lambdaIdn)
        print()

    characteristicMatrix = MatrixSubtract(matrixA=A, matrixB=lambdaIdn)

    if verbose:

        print(
            f"Characteristic Matrix ({pretty(Symbol("A") - lamda * Symbol(f"I_{A.numRows}"))}):"
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

    realEigenValuesRaw = characteristicPoly.real_roots()

    realEigenValues = set()
    complexEigenValues = set()

    for eigenValueRaw in eigenValuesRaw:

        if eigenValueRaw not in realEigenValuesRaw:

            complexEigenValue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            complexEigenValues.add(complexEigenValue)

        else:

            realEigenValue = Eigenvalue(
                value=eigenValueRaw, multiplicity=eigenValuesRaw.count(eigenValueRaw)
            )

            realEigenValues.add(realEigenValue)

    realEigenValues = sorted(realEigenValues, key=lambda x: x.value)
    complexEigenValues = sorted(complexEigenValues, key=lambda x: x.value)

    for i, realEigenValue in enumerate(realEigenValues):

        realEigenValue.index = i

    for i, complexEigenValue in enumerate(complexEigenValues):

        complexEigenValue.index = f"{i}i"

    if verbose:

        print("Real Eigenvalues:")

        for eigenValue in realEigenValues:

            print(f"  {eigenValue}")

    if verbose and len(complexEigenValues) > 0:

        print("Complex Eigenvalues:")

        for complexEigenValue in complexEigenValues:

            print(f"  {complexEigenValue}")

        print()


A = FloatMatrix(np.array([[4, 2], [3, 4]]))


eigenValues = GetEigenvalues(A=A, verbose=True)
