from sympy import Symbol

from . import *


def Eigenvalues(A: FloatMatrix):

    if A.numRows != A.numCols:

        raise Exception("Matrix must be square")

    lambda_ = Symbol("lambda")

    idn = Idn(A.numRows)

    idn = ScalarMultiply(scalar=lambda_, matrix=idn)

    print(idn)


A = Matrix(2, 2)
print(A)

Eigenvalues(A=A)

print(A)
