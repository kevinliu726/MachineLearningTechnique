import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
# data pre-processing
train = []
C_array = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
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
NUM = x.shape[0]
y[y!=8] = -1
y[y==8] = 1
plot_x, plot_y = [-5, -3, -1, 1, 3], []
from sklearn import svm
for c in C_array:
	clf = svm.SVC(C = c,degree = 2, gamma = 1, coef0 = 1, kernel='poly', verbose=False)
	clf.fit(x, y)
	ans = clf.predict(x)
	error = 1 - np.sum(np.asarray(list(map(int, ans == y)))) / NUM
	plot_y.append(error)
	print("Ein---->", error)
plot_x, plot_y = np.asarray(plot_x), np.asarray(plot_y)
plt.plot(plot_x, plot_y, marker = 'o', mec = 'red', mfc = 'red', linestyle = '--')
plt.xlabel(r'$log_{10} C$')
plt.ylabel(r'$E_{in}$')
plt.savefig('12.png')
plt.show()
