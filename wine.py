from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. 와인 데이터셋 로드
wine = datasets.load_wine()

X = wine.data
y = wine.target

# (80:20)으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print("학습 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)

# 학습 (Parameter 변경 실험: weights='distance' 적용)
# 거리가 가까운 이웃에게 더 큰 가중치를 부여
knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(X_train, y_train)

# 평가
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(f"모델 정확도(Accuracy): {scores:.4f}")

# 예측
# 와인 데이터셋의 클래스 이름은 class_0, class_1, class_2
classes = {0: 'class_0', 1: 'class_1', 2: 'class_2'}

# 전혀 보지 못한 새로운 데이터를 제시
x_new = [
    [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
    [12.37, 1.17, 1.92, 19.6,  78.0, 2.11, 2.00, 0.27, 1.04, 4.68, 1.12, 3.48,  510.0]
]

y_predict = knn.predict(x_new)
print(f"첫 번째 샘플 예측 결과: {classes[y_predict[0]]}")
print(f"두 번째 샘플 예측 결과: {classes[y_predict[1]]}")
