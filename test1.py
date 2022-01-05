import numpy as np


def viterbi(stateSpace, initialProbs, observations, transitionMatrix, emissionMatrix):
  numberOfStates = len(stateSpace)
  numberOfObservations = len(observations)

  probabilities = [
       [0.0] * numberOfObservations for i in range(numberOfStates)]

  # first step
  for i in range(numberOfStates):
      probabilities[i][0] = emissionMatrix[i][observations[0] -
          1] * initialProbs[i]

  # rest of the steps
  for j in range(1, numberOfObservations):
      for i in range(numberOfStates):
          for k in range(numberOfStates):
              probabilities[i][j] = max(probabilities[i][j],
                                        probabilities[k][j-1] *
                                        emissionMatrix[i][observations[j] - 1] *
                                        transitionMatrix[k][i])

  print(probabilities)
    
    
viterbi(["H", "C"], [0.8, 0.2], [3, 1, 3], [[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])
