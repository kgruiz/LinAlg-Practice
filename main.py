import os
import random

import numpy as np
from tqdm import tqdm

from matrix import FloatMatrix, Matrix
from multiply import Multiply
from vectorDot import VectorDot


def testMatrix(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 1,
    min_: int = 0,
    max_: int = 100,
    verbose: bool = True,
):

    if verbose:

        print(f"Making A and B\n")

    a = Matrix(Arows, Acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")


def testVectorDot(
    Arows: int = 4,
    Acols: int = 1,
    Brows: int = 1,
    Bcols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = True,
):

    a = Matrix(Arows, Acols, min_, max_)

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

        print(f"B:\n{b}")

    if verbose:

        print(f"Calculating Dot Product of A and B\n")

    c = VectorDot(a, b)

    if isinstance(c, str):

        print(f"{c}\n")

        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")

        return False

    if verbose:

        print(f"C:\n{c}")

    a = a.reshape(-1)
    b = b.reshape(-1)

    d = np.dot(a, b)

    correct = c == d

    if not correct:

        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")

        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")

        return True


def testMultiply(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 2,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = True,
) -> bool:

    a = Matrix(Arows, Acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")

        print(f"Multiplying A and B\n")

    c = Multiply(matrixA=a, matrixB=b)

    if isinstance(c, str):

        print(f"{c}\n")

        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")

        return False

    if verbose:

        print(c)

    d = np.matmul(a, b)

    correct = c.shape == d.shape

    for row in range(c.shape[0]):

        for col in range(c.shape[1]):

            if not correct:

                break

            correct = c[row][col] == d[row][col]

    if not correct:

        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")

        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")

        return True


def randomTestVectorDot(
    minSize: int = -100, maxSize: int = 100, verbose: bool = True
) -> bool:

    vectorLength = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:

        Arows = vectorLength
        Acols = 1

    else:

        Arows = 1
        Acols = vectorLength

    if random.randrange(0, 2) == 0:

        Brows = vectorLength
        Bcols = 1

    else:

        Brows = vectorLength
        Bcols = 1

    # # Ensures error checking is correct, but rarely runs
    # if random.randrange(0, 100) == 99:

    #     if random.randrange(0, 2) == 0:

    #         Arows = vectorLength
    #         Acols = random.randrange(0, vectorLength)

    #     else:

    #         Brows = vectorLength
    #         Bcols = random.randrange(0, vectorLength)

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


def randomTestMultiply(
    minSize: int = -100, maxSize: int = 100, verbose: bool = True
) -> bool:

    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Acols
    Bcols = random.randint(0, maxSize)

    # # Ensures error checking is correct, but rarely runs
    # if random.randrange(0, 100) == 99:

    #     Arows = random.randint(0, maxSize)
    #     Acols = random.randint(0, maxSize)
    #     Brows = random.randint(0, maxSize)
    #     Bcols = random.randint(0, maxSize)

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


if __name__ == "__main__":

    minSize = -100
    maxSize = 100

    for _ in tqdm(range(1000)):

        testResult = randomTestVectorDot(
            minSize=minSize, maxSize=maxSize, verbose=False
        )

        if not testResult:

            print("VectorDot")

            break

        testResult = randomTestMultiply(minSize=minSize, maxSize=maxSize, verbose=False)

        if not testResult:

            print("Multiply")

            break
