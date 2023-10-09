import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# raw_data = pandas.read_csv('perceptron/cars.csv')

# data = raw_data.apply(LabelEncoder().fit_transform)
# data = data[['buying','maint','doors','persons','lug_boot','safety','acceptability']].values

# train, test = train_test_split(data, test_size=0.3, shuffle=True)

# xTrain = train[:,:6]
# yTrain = train[:,6]
# xTest = test[:,:6]
# yTest = test[:,6]

try:
    raw_data = pandas.read_csv('../perceptron/mushrooms.csv')
except:
    raw_data = pandas.read_csv('mushrooms.csv')

## Đọc dữ liệu
data = raw_data.apply(LabelEncoder().fit_transform)
## Loại bỏ tiêu đề
data = data[['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']].values

## Tách dữ liệu 
train, test = train_test_split(data, test_size=0.3, shuffle=True)

## Nhãn là cột đầu tiên, bộ dữ liệu là 9 cột kế tiếp
xTrain, yTrain = train[:,1:], list(train[:,0])            ## Dữ liệu train
xTest, yTest = test[:,1:], list(test[:,0])                ## Dữ Liệu test

tree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5,min_samples_split=10)
tree.fit(xTrain, yTrain)

yPredict = tree.predict(xTest)

count = 0                                                   ## Tạo một biến đếm
for i, v in enumerate(yPredict):                            ###-------------------------###
    if v == yTest[i]:                                       ### Đếm số lần dự đoán đúng ###
        count += 1                                          ###-------------------------###

## In kết quả
print('Du doan dung {0}/{1} = {2}'.format(count, len(yTest), count/len(yTest)))

# print(tree.get_depth())
# print(accuracy_score(yTest, ypred))