import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
# data pre-processing
train = []
C_array = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

with open("features.train", 'r') as f:
	for line in f:
		train.append([float(i) for i in line.split()])

train = np.asarray(train)

x, y = [], []
for i in train:
	digit, inten, symm = i[0], i[1], i[2]
	x.append([inten, symm])
	y.append(digit)
x, y = np.asarray(x), np.asarray(y)
y[y!=0] = -1
y[y==0] = 1
plot_x, plot_y = [-3, -2, -1, 0, 1], []

def K(x1, x2):
    tmp = np.sqrt(np.dot(x1-x2, x1-x2))**2
    return np.exp(-80*tmp)

def distance(clf):
	w = 0.0
	n = len(clf.support_)
	for i in range(n):
		for j in range(n):
			x1 = x[clf.support_[i]]
			x2 = x[clf.support_[j]]
			w += clf.dual_coef_[0][i] * clf.dual_coef_[0][j] * K(x1,x2)
	
	w = np.sqrt(w)
	return 1/w

from sklearn import svm
for c in C_array:
	print(c)
	clf = svm.SVC(C = c, gamma = 80, kernel = 'rbf')
	clf.fit(x, y)
	plot_y.append(distance(clf))

plot_x, plot_y = np.asarray(plot_x), np.asarray(plot_y)
plt.plot(plot_x, plot_y, marker = 'o', mec = 'red', mfc = 'red', linestyle = '--')
plt.xlabel(r'$log_{10} C$')
plt.ylabel(r'$Distance$')
plt.savefig('14.png')
plt.show()
