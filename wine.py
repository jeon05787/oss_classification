from sklearn import datasets
# 1. 와인 데이터셋 로드
wine = datasets.load_wine()
print(wine)

from sklearn.model_selection import train_test_split
X = wine.data
y = wine.target

# (80:20)으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(X_train.shape)
print(X_test.shape)

# 학습
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# 평가
y_pred = knn.predict(X_test)
from sklearn import metrics

scores = metrics.accuracy_score(y_test, y_pred)
print(scores)

# 예측
# 와인 데이터셋의 클래스 이름은 class_0, class_1, class_2
classes = {0: 'class_0', 1: 'class_1', 2: 'class_2'}

# 전혀 보지 못한 새로운 데이터를 제시
# 와인은 특성이 13개이므로, 샘플 하나당 13개의 숫자를 넣어야함
x_new = [
    [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
    [12.37, 1.17, 1.92, 19.6,  78.0, 2.11, 2.00, 0.27, 1.04, 4.68, 1.12, 3.48,  510.0]
]

y_predict = knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])
