import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [1, 50],
    [2, 55],
    [9, 95]
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

prediction = model.predict([[study_hours, attendance]])

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")