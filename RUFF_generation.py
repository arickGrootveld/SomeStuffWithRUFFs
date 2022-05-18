### In this file you can find several helper functions, along with the much more important
### generateRUFF(n,m,d,k,alpha), which generates a binary RUFF with those parameters
### using random vectors

import numpy as np
from numpy.random import default_rng

## Helper functions
# Simple helper function to generate the cartesian product of some set of arrays
# of numbers
# This one stolen from: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# Function to covnert an array to a column vector
def fromArrayToColVec(targArray):
    columnVec = (targArray)[np.newaxis].T
    return(columnVec)

# Function to take the union of all columns of the input matrix
def unionOfColumns(matrix):
    unionColumn = fromArrayToColVec(matrix[:,0])
    for i in range(1, matrix.shape[1]):
        unionColumn = unionColumn | fromArrayToColVec(matrix[:,i])
    return(unionColumn)
    
# Function that will generate a random binary vector with "length"
# elements, and with a cardinality of "numOnes"
def generateRandomBinaryVector(length, numOnes=10, rng=default_rng()):
    indexVector = np.arange(length, dtype=np.int64)
    randomOneIndexes = rng.choice(indexVector, size=numOnes, replace=False)    
    randBinVec = np.zeros(length, dtype=np.int64)    
    randBinVec[randomOneIndexes] = 1    
    return(randBinVec)
    
# Function to check whether a given matrix is an (n,m,d,k,alpha)-RUFF
def RUFFTest(matrix, k=2, alpha=0.5):
    # Determining n,m, and d
    [m,n] = matrix.shape
    # Going through each column of the array, and making sure they
    # each have cardinality d
    d = np.count_nonzero(matrix[:,0])
    for i in range(1,n):
        cardOfTargColumn = np.count_nonzero(matrix[:,i])
        if(d != cardOfTargColumn):
            raise Exception('Cardinality of column 0 and column ' + str(i) + ' do not match')
    # Going through each set of columns, and if its not comparing the same column
    # with itself, we check to see if the intersection of the union of the 
    # target columns is below the required threshold
    possibleColumnNumbers = np.arange(n)
    arrayOfSetsOfArrays = np.tile(possibleColumnNumbers, (k,1))
    columnSets = cartesian_product(*arrayOfSetsOfArrays)
    for targColumnSet in columnSets:
        # Spliting the elements into the test column, and the rest of the columns
        testColumn = targColumnSet[0]
        restColumns = targColumnSet[1:]
        # Only checking if the test column is not one of the columns of the test matrix
        if(testColumn not in restColumns):
            testColumnActual = fromArrayToColVec(matrix[:,testColumn])
            restColumnsActual = unionOfColumns(matrix[:, restColumns])
            cardOfIntersection = np.count_nonzero(testColumnActual & restColumnsActual)
            if(cardOfIntersection >= (alpha * d)):
                return(False)
    # If we got outside the for loop without returning false, then it must be an RUFF
    return(True)


## Main function of this file

# Function that will generate an RUFF of the given parameters
def generateRUFF(n,m,d,k,alpha):
    # The process that we will be using to generate an RUFF of the provided
    # parameters will be to generate random vectors and concatenate them to
    # a matrix if the matrix is still an RUFF with the vector concatenated

    # Step 1: Generate a random binary vector to start from
    targRuff = fromArrayToColVec(generateRandomBinaryVector(length=m, numOnes=d))
    fullRUFF_generated = False
    while(not fullRUFF_generated):
        newVec = fromArrayToColVec(generateRandomBinaryVector(length=m, numOnes=d))
        newMatrix = np.concatenate((targRuff, newVec), axis=1)
        isRUFF = RUFFTest(newMatrix, k=k, alpha=alpha)
        if(isRUFF):
            targRuff = newMatrix
            if(targRuff.shape[1] == n):
                fullRUFF_generated = True
    
    return(targRuff)