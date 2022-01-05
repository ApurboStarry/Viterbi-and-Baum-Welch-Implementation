import numpy as np
import math


def getInitialProbs(transitionMatrix):
    transitionMatrix = np.array(transitionMatrix)
    evals, evecs = np.linalg.eig(transitionMatrix.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    
    return stationary


def getEmissionProbability(x, state, means, variances):
    mu = means[state]
    sigmaSquared = variances[state]

    return (1.0 / (math.sqrt(2*math.pi*sigmaSquared))) * math.exp(-0.5*((x - mu) ** 2 / sigmaSquared))


def getParametersFromFile(parameterFileName):
    with open(parameterFileName) as f:
        n = int(next(f))
        transitionMatrix = []
        for i in range(n):
            row = [float(x) for x in next(f).split()]
            transitionMatrix.append(row)

        means = [float(x) for x in next(f).split()]
        variances = [float(x) for x in next(f).split()]

    return n, transitionMatrix, means, variances


def viterbi(stateSpace, initialProbs, observations, transitionMatrix, means, variances):
    numberOfStates = len(stateSpace)
    numberOfObservations = len(observations)

    probabilities = [
        [0.0] * numberOfObservations for i in range(numberOfStates)]

    # first column
    for i in range(numberOfStates):
        probabilities[i][0] = getEmissionProbability(
            observations[0], i, means, variances) * initialProbs[i]

    # rest of the columns
    for j in range(1, numberOfObservations):
        for i in range(numberOfStates):
            for k in range(numberOfStates):
                probabilities[i][j] = max(probabilities[i][j],
                                          probabilities[k][j-1] *
                                          getEmissionProbability(observations[j], i, means, variances) *
                                          transitionMatrix[k][i])
                
    result = []
    for j in range(numberOfObservations):
        maxStateValue = 0
        maxState = 0
        for i in range(numberOfStates):
            if maxStateValue < probabilities[i][j]:
                maxStateValue = probabilities[i][j]
                maxState = i
                
        result.append(stateSpace[maxState])
        
    return result


def getObservationsFromDataFile(dataFileName):
    observations = []
    with open(dataFileName) as f:
        for line in f:
            observations.append(float(line))

    return observations


def writeToFile(fileName, dataArray):
    file = open(fileName, 'w')
    for line in dataArray:
        file.write(line + "\n")
    file.close()


def testViterbi(dataFileName, parameterFileName):
    n, transitionMatrix, means, variances = getParametersFromFile(
        parameterFileName)
    stateSpace = ["El Nino", "La Nina"]
    initialProbs = getInitialProbs(transitionMatrix)
    observations = getObservationsFromDataFile(dataFileName)
    viterbiResult = viterbi(stateSpace, initialProbs,
            observations, transitionMatrix, means, variances)
    writeToFile("viterbi-result.txt", viterbiResult)


testViterbi("data.txt", "parameters.txt.txt")
