#!/usr/bin/python
#
# Author: Alex von Brandenfels
import sys
import copy

class Perceptron(object):
    def __init__(self, varnames):
        self._bias = 0.0
        self._vectorSize = len(varnames)
        self._weights = [0.0 for i in range(self._vectorSize)]
        self._varnames = copy.copy(varnames)
    
    """
    Returns the activation for the given data point
    """
    def getActivation(self, dataVector):
        result = self._bias
        for i in range(self._vectorSize):
            result += self._weights[i] * dataVector[i]
        return result
    
    """
    Classify a data point
    """
    def predict(self, dataVector):
        if self.getActivation(dataVector) > 0:
            return 1
        return -1
    
    """
    Perform a single update based on a single data point
    Returns True if already predicted correctly
    """
    def learn(self, dataRow):
        dataVector, desiredOutput = dataRow
        isCorrect = self.getActivation(dataVector) * desiredOutput > 0
        if not isCorrect:
            # Classified wrong
            self._bias += desiredOutput
            for i in range(self._vectorSize):
                self._weights[i] += desiredOutput * dataVector[i]
        return isCorrect
    
    """
    Train from a data set. Ends after convergence or after max_iters iterations
    """
    def train(self, dataSet, max_iters=100):
        dsize = len(dataSet)
        for i in range(max_iters):
            done = True
            for row in dataSet:
                if self.learn(row) == False:
                    # Data point was classified incorrectly
                    done = False
            if done:
                break
    
    """
    Test on a data set and return the fraction that was correctly classified
    """
    def testAccuracy(self, dataSet):
        if len(dataSet) == 0:
            raise ValueError("Data set can't be empty")
        correct = 0
        for row in dataSet:
            if self.predict(row[0]) == row[1]:
                correct += 1
        return float(correct) / len(dataSet)
    
    """
    Returns a string representation of the perceptron
    """
    def __repr__(self):
        result = ["bias " + str(self._bias)]
        for i in range(self._vectorSize):
            result.append("\n{} {}".format(self._varnames[i], self._weights[i]))
        return "".join(result)
        
   
"""
Parses a csv, where the label is the last column   
"""
def read_data(filename):
    with open(filename, 'r') as dataFile:
        data = []
        varnames = dataFile.readline().strip().split(",")
        for row in dataFile:
            vector = list(map(float, row.split(",")))
            features = vector[:-1]
            label = vector[-1]
            if label == 0:
                label = -1
            data.append((features,label))
        return (data, varnames[:-1])

def main(argv):
    if (len(argv) != 3):
        print('Usage: perceptron.py <train> <test> <max_iters>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    test = read_data(argv[1])[0]
    max_iters = int(argv[2])

    # Train model
    print("Training...")
    p = Perceptron(varnames)
    p.train(train, max_iters)
    
    # Test model
    print("Testing...")
    accuracy = p.testAccuracy(test)
    print("Classifier had {} accuracy on the test data".format(accuracy))

if __name__ == "__main__":
    main(sys.argv[1:])
