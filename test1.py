import numpy as np

# determination of stationary matrix
# transitionMatrix = np.array([[0.5, 0.3, 0.2],
#                              [0.4, 0.2, 0.4],
#                              [0.0, 0.3, 0.7]])
transitionMatrix = np.array([[0.2, 0.6, 0.2],
                             [0.3, 0.0, 0.7],
                             [0.5, 0.0, 0.5]])
rhs = np.array([1, 1, 1]).reshape(1, 3)

# print(transitionMatrix.transpose())
# print(np.matmul(rhs, np.linalg.inv(transitionMatrix)))

evals, evecs = np.linalg.eig(transitionMatrix.T)
evec1 = evecs[:, np.isclose(evals, 1)]
evec1 = evec1[:, 0]
stationary = evec1 / evec1.sum()
stationary = stationary.real
print(stationary)


def getInitialProbs(transitionMatrix):
    n = transitionMatrix.shape[0]
    rhs = np.array(np.ones(n)).reshape(1, 3)

    return np.matmul(rhs, np.linalg.inv(transitionMatrix))

