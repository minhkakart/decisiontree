import math

class Node:
    def __init__(self, label: str, attributes = {}) -> None:
        self.label = label
        self.attributes = attributes
    def predict(self, data):
        if len(self.attributes) == 0:
            return self.label
        reduceData = dict(data)
        del reduceData[self.label]
        res = self.attributes[data[self.label]].predict(reduceData)
        return res
    
class MyDecisionTreeClassifier():
    def __init__(self, title: list, x_train, y_train) -> None:
        if len(title) != len(x_train[0]) or len(x_train) != len(y_train):
            raise ValueError('Parameters is not valid!')
        self.title = title
        self.yTrain = y_train
        self.xTrain = []
        if not str(type(x_train[0])) == "<class 'dict'>":
            for i in x_train:
                record = {}
                for index, val in enumerate(i):
                    record[self.title[index]] = val
                self.xTrain.append(record)
        else:
            self.xTrain = x_train

    def entropy(self, data, className, title):
        att = set(map(lambda x: x[title], data))
        entroPoint = 0
        for i in att:
            yclass = []
            for index, val in enumerate(data):
                if val[title] == i:
                    yclass.append(className[index])
            numberOfClass = set(yclass)
            totalClass = len(yclass)
            subPoint = 0
            for j in numberOfClass:
                count = yclass.count(j)
                subPoint += -(count/totalClass)*math.log(count/totalClass)
            entroPoint += totalClass/len(data)*subPoint
        return entroPoint

    def buildTree(self, xData, yClass, labels: list):
        className = list(set(yClass))
        if len(className) == 1:
            return Node(className[0])
        if len(labels) == 0:
            count = []
            for i in className:
                count.append(yClass.count(i))
            return Node(str(className[count.index(max(count))]))
        
        entropyPoint = []
        for label in labels:
            entropyPoint.append(self.entropy(xData, yClass, label))
        
        minPointStr = labels.__getitem__(entropyPoint.index(min(entropyPoint)))
        listAtt = set(map(lambda x: x[minPointStr], xData))
        nodeAtt = {}
        labelsReduce = list(labels)
        labelsReduce.remove(minPointStr)
        for i in listAtt:
            subData = []
            subLabels = []
            for index, val in enumerate(xData):
                if val[minPointStr] == i:
                    subData.append(val)
                    subLabels.append(yClass[index])
            nodeAtt[i] = self.buildTree(subData, subLabels, labelsReduce)
        return Node(minPointStr, nodeAtt)

    def fit(self):
        self.tree = self.buildTree(self.xTrain, self.yTrain, self.title)

    def predict(self, data: list):
        if not str(type(data[0])) == "<class 'dict'>":
            newData = []
            for i in data:
                record = {}
                for index, val in enumerate(i):
                    record[self.title[index]] = val
                newData.append(record)
        else:
            newData = data
        return list(map(lambda x: self.tree.predict(x), newData))
{
# data = [
#     {'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'wind': 'weak'},            ##no
#     {'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'wind': 'strong'},          ##no
#     {'outlook': 'overcast', 'temp': 'hot', 'humidity': 'high', 'wind': 'weak'},         ##yes
#     {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'wind': 'weak'},           ##yes
#     {'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'wind': 'weak'},         ##yes
#     {'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'wind': 'strong'},       ##no
#     {'outlook': 'overcast', 'temp': 'cool', 'humidity': 'normal', 'wind': 'strong'},    ##yes
#     {'outlook': 'sunny', 'temp': 'mild', 'humidity': 'high', 'wind': 'weak'},           ##no
#     {'outlook': 'sunny', 'temp': 'cool', 'humidity': 'normal', 'wind': 'weak'},         ##yes
#     {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'normal', 'wind': 'weak'},         ##yes
#     {'outlook': 'sunny', 'temp': 'mild', 'humidity': 'normal', 'wind': 'strong'},       ##yes
#     {'outlook': 'overcast', 'temp': 'mild', 'humidity': 'high', 'wind': 'strong'},      ##yes
#     {'outlook': 'overcast', 'temp': 'hot', 'humidity': 'normal', 'wind': 'weak'},       ##yes
#     {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'wind': 'strong'}          ##no
#         ]
# yLabels = ['no', 'no','yes', 'yes','yes', 'no','yes', 'no','yes', 'yes','yes', 'yes','yes', 'no']
# tree = MyDecisionTreeClassifier(['outlook', 'temp', 'humidity', 'wind'], data, yLabels)
# tree.fit()
# print(tree.predict(data))
# print(tree.xTrain)

# tree = Node('outlook', 
#             {'sunny': Node('humidity', 
#                               {'high': Node('no'), 
#                               'normal': Node('yes')}), 
#             'overcast': Node('yes'), 
#             'rainy': Node('wind', 
#                               {'weak': Node('yes'), 
#                           '   strong': Node('no')})})

# print(tree.predict(data))
}
