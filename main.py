import pandas
# from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Tree import MyDecisionTreeClassifier

try:
    raw_data = pandas.read_csv('perceptron/mushrooms.csv')
except:
    raw_data = pandas.read_csv('mushrooms.csv')

## Đọc dữ liệu
data = raw_data.apply(LabelEncoder().fit_transform)
## Loại bỏ tiêu đề
data = data[['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color']].values

## Tách dữ liệu 
train, test = train_test_split(data, test_size=0.3, shuffle=True)

## Nhãn là cột đầu tiên, bộ dữ liệu là 9 cột kế tiếp
xTrain, yTrain = train[:,1:10].tolist(), train[:,0].tolist()          ## Dữ liệu train
xTest, yTest = test[:,1:10].tolist(), test[:,0].tolist()             ## Dữ Liệu test
# print((xTest))

title = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color']

DecisionTree = MyDecisionTreeClassifier(title=title, x_train=xTrain, y_train=yTrain, criterion='gini')
DecisionTree.fit()
yPredict = DecisionTree.predict(xTest)                                 ## Dự đoán tập test
count = 0                                                   ## Tạo một biến đếm
for i, v in enumerate(yPredict):                            ###-------------------------###
    if v == yTest[i]:                                       ### Đếm số lần dự đoán đúng ###
        count += 1                                          ###-------------------------###

## In kết quả
print('Du doan dung {0}/{1} = {2}'.format(count, len(yTest), count/len(yTest)))