import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

train = []
with open('features.train', 'r') as f:
    for line in f:
        train.append([float(i) for i in line.split()])

train = np.array(train)
x = []
y = []

for i in train:
    digit, intensity, symmetry = i[0], i[1], i[2]
    if digit == 0:
        y.append(1)
    else:
        y.append(-1)
    x.append([intensity, symmetry])

x = np.array(x)
y = np.array(y)

C_a = np.array([10**i for i in [-3, -2, -1, 0, 1]])
def K(x1, x2):
    tmp = np.sqrt(np.dot(x1-x2, x1-x2))**2
    return np.exp(-80*tmp)
def cal_distance(clf):
    w = 0.0
    for i in range(len(clf.support_)):
        for j in range(len(clf.support_)):
            x1 = x[clf.support_[i]]
            x2 = x[clf.support_[j]]
            w += clf.dual_coef_[0][i]*clf.dual_coef_[0][j]*K(x1, x2)

    w = np.sqrt(w)
    return 1/w

distance = []
for i in C_a:
    clf = svm.SVC(C = i, gamma = 80, kernel='rbf')
    clf.fit(x, y)
    distance.append(cal_distance(clf))

plt.semilogx(C_a, distance, '--o')
plt.ylabel('distance', fontsize = 15)
plt.xlabel('C', fontsize = 15)
for i, j in zip(C_a, distance):
    plt.text(i, j, str(round(j, 6)))
plt.savefig('tes.png')
plt.show()
