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


def testViterbi(dataFileName, parameterFileName, stateSpace):
    n, transitionMatrix, means, variances = getParametersFromFile(
        parameterFileName)
    
    initialProbs = getInitialProbs(transitionMatrix)
    observations = getObservationsFromDataFile(dataFileName)
    viterbiResult = viterbi(stateSpace, initialProbs,
            observations, transitionMatrix, means, variances)
    writeToFile("viterbi-result.txt", viterbiResult)
    
    
class BaumWelch:
    def __init__(self, dataFileName, parameterFileName, stateSpace):
        self.observations = getObservationsFromDataFile(dataFileName)
        n, transitionMatrix, means, variances = getParametersFromFile(
            parameterFileName)
        self.n = n
        self.transitionMatrix = transitionMatrix
        self.means = means
        self.variances = variances
        self.stateSpace = stateSpace
        self.initialProbs = getInitialProbs(transitionMatrix)
        
        numberOfObservations = len(self.observations)
        numberOfStates = len(self.stateSpace)
        self.forwardTable = [[-1] * numberOfStates for i in range(numberOfObservations)] # numberOfObservations * numberOfStates
        self.backwardTable = [[-1] * numberOfStates for i in range(numberOfObservations)] # numberOfObservations * numberOfStates
        self.piStarTable = [[-1] * numberOfStates for i in range(numberOfObservations)] # numberOfObservations * numberOfStates
        self.piStarStarTable = [[[-1] * numberOfStates for i in range(numberOfStates)] for i in range(numberOfObservations)] # numberOfObservations * numberOfStates * numberOfStates
        
        
    def forwardAlgorithm(self):
        length = len(self.observations)
        numberOfStates = len(self.stateSpace)
        
        for i in range(numberOfStates):
            self.forwardTable[0][i] = self.initialProbs[i] * getEmissionProbability(self.observations[0], i, self.means, self.variances)
            
        for i in range(1, length):
            for j in range(numberOfStates):
                summation = 0
                for k in range(numberOfStates):
                    summation += self.forwardTable[i-1][k] * self.transitionMatrix[k][j] * getEmissionProbability(self.observations[i], j, self.means, self.variances)
                    
                self.forwardTable[i][j] = summation
                        
        
    def backwardAlgorithm(self):
        length = len(self.observations)
        numberOfStates = len(self.stateSpace)
        
        for i in range(numberOfStates):
            self.backwardTable[length - 1][i] = 1
        
        for i in range(length - 2, -1, -1):
            for j in range(numberOfStates):
                summation = 0
                for k in range(numberOfStates):
                    summation += self.backwardTable[i+1][k] * self.transitionMatrix[j][k] * getEmissionProbability(self.observations[i], j, self.means, self.variances)
                    
                self.backwardTable[i][j] = summation
    
    
    def fillPiStarTable(self):
        length = len(self.observations)
        numberOfStates = len(self.stateSpace)
        
        for i in range(length):
            for j in range(numberOfStates):
                self.piStarTable[i][j] = self.forwardTable[i][j] * self.backwardTable[i][j]

    
    def fillPiStarStarTable(self):
        # print(len(self.piStarStarTable), len(self.piStarStarTable[0]), len(self.piStarStarTable[0][0]))
        for i in range(len(self.piStarStarTable) - 1):
            for j in range(len(self.piStarStarTable[0])):
                for k in range(len(self.piStarStarTable[0][0])):
                    self.piStarStarTable[i][j][k] = self.forwardTable[i][k] * self.transitionMatrix[j][k] * getEmissionProbability(self.observations[i + 1], k, self.means, self.variances) * self.backwardTable[i+1][k]


    def updateTransitionMatrix(self):
        # print(self.piStarStarTable[0])
        numberOfStates = len(self.stateSpace)
        numberOfObservations = len(self.observations)
        
        for i in range(numberOfStates):
            for j in range(numberOfStates):
                summation = 0
                for k in range(numberOfObservations):
                    summation += self.piStarStarTable[k][i][j]
                    
                self.transitionMatrix[i][j] = summation


    def updateMeans(self):
        numberOfStates = len(self.stateSpace)
        numberOfObservations = len(self.observations)
        
        for i in range(numberOfStates):
            numerator = 0
            denominator = 0
            
            for j in range(numberOfObservations):
                numerator += self.piStarTable[j][i] * self.observations[j]
                denominator += self.piStarTable[j][i]
                
            if denominator > 0: self.means[i] = numerator / denominator
            
            
    def updateVariances(self):
        numberOfStates = len(self.stateSpace)
        numberOfObservations = len(self.observations)

        for i in range(numberOfStates):
            numerator = 0
            denominator = 0

            for j in range(numberOfObservations):
                numerator += self.piStarTable[j][i] * (self.observations[j] - self.means[i]) ** 2
                denominator += self.piStarTable[j][i]

            if denominator > 0:
                self.means[i] = numerator / denominator


    def initializeBaumWelchParameters(self):
        import random
        numberOfStates = len(self.stateSpace)
        
        newTransitionMatrix = []

        for i in range(numberOfStates):
            tempTransitionMatrix = []
            summation = 0
            for j in range(numberOfStates - 1):
                transitionProb = random.uniform(0, 1 - summation)
                summation += transitionProb
                tempTransitionMatrix.append(transitionProb)
            tempTransitionMatrix.append(1 - summation)
            newTransitionMatrix.append(tempTransitionMatrix)
            
        self.transitionMatrix = newTransitionMatrix
    
    
    def writeParametersToFile(self):
        numberOfStates = len(self.stateSpace)
        file = open("Baum Welch Parameters", 'w')
        file.write(str(numberOfStates) + "\n")
        # for line in dataArray:
        #     file.write(line + "\n")
            
        for i in range(numberOfStates):
            for j in range(numberOfStates):
                file.write(str(self.transitionMatrix[i][j]) + " ")
            file.write("\n")
        
        for i in range(numberOfStates):
            file.write(str(self.means[i]) + " ")
        file.write("\n")
        
        for i in range(numberOfStates):
            file.write(str(self.variances[i]) + " ")
        file.write("\n")
            
        file.close()
    
    
    def baumWelchAlgo(self):
        self.initializeBaumWelchParameters()
        
        for i in range(10):
            self.forwardAlgorithm()
            self.backwardAlgorithm()
            self.fillPiStarTable()
            self.fillPiStarStarTable()

            self.updateTransitionMatrix()
            self.updateMeans()
            self.updateVariances()
            
        self.writeParametersToFile()
        


stateSpace = ["El Nino", "La Nina"]
# testViterbi("data.txt", "parameters.txt.txt", stateSpace)
baumWelch = BaumWelch("data.txt", "parameters.txt.txt", stateSpace)
# baumWelch.forwardAlgorithm()
# baumWelch.backwardAlgorithm()
baumWelch.baumWelchAlgo()
