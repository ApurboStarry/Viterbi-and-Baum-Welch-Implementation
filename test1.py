def initializeBaumWelchParameters():
    import random
    numberOfStates = 2

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

    print(newTransitionMatrix)


initializeBaumWelchParameters()