# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:31:08 2022

@author: aegrootv
"""

import numpy as np
from numpy.random import default_rng

# Simulation parameters
alpha = 0.6
k = 3
m = 30
n = 10
d = 3

seed = 420

rng = default_rng(seed)

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
def generateRandomBinaryVector(length, numOnes=10):
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
    
    # TODO: Check that the above is working as expected
    possibleColumnNumbers = np.arange(n)
    arrayOfSetsOfArrays = np.tile(possibleColumnNumbers, (k,1))
    columnSets = cartesian_product(*arrayOfSetsOfArrays)
    
    # TODO: Check that the code is working here as well
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



# # Testing that the function actually works
# x = np.identity(5,dtype=np.int64)

# print(RUFFTest(x, k=3, alpha=0.5))
    

## Code to generate an RUFF by randomly generating vectors till the RUFF of the required parameters is generated

# Step 1: Generate a random binary vector to start from
targRuff = fromArrayToColVec(generateRandomBinaryVector(length=m, numOnes=d))

# Step 2: Generate new binary vectors, and concatenate them into RUFF we have
# thus far, and make sure that it is still an RUFF with the required properties

fullRUFF_generated = False
while(not fullRUFF_generated):
    

    newVec = fromArrayToColVec(generateRandomBinaryVector(length=m, numOnes=d))
    
    newMatrix = np.concatenate((targRuff, newVec), axis=1)
    
    isRUFF = RUFFTest(newMatrix, k=k, alpha=alpha)
    
    if(isRUFF):
        targRuff = newMatrix
        if(targRuff.shape[1] == n):
            fullRUFF_generated = True
            print(targRuff)
    




    