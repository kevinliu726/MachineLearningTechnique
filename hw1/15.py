import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
# data pre-processing
train = []
test = []
G_array = [1e0, 1e1, 1e2, 1e3, 1e4]

with open("features.train", 'r') as f:
	for line in f:
		train.append([float(i) for i in line.split()])

train = np.asarray(train)
with open("features.test", 'r') as f:
	for line in f:
		test.append([float(i) for i in line.split()])

test = np.asarray(train)

x, y = [], []
for i in train:
	digit, inten, symm = i[0], i[1], i[2]
	x.append([inten, symm])
	y.append(digit)
x, y = np.asarray(x), np.asarray(y)
y[y!=0] = -1
y[y==0] = 1

xt, yt = [], []
for i in test:
	digit, inten, symm = i[0], i[1], i[2]
	xt.append([inten, symm])
	yt.append(digit)
xt, yt = np.asarray(xt), np.asarray(yt)
yt[yt!=0] = -1
yt[yt==0] = 1

plot_x, plot_y = [0, 1, 2, 3, 4], []

def K(x1, x2):
    tmp = np.sqrt(np.dot(x1-x2, x1-x2))**2
    return np.exp(-80*tmp)


Eout = []
from sklearn import svm
for g in G_array:
	clf = svm.SVC(C = 0.1, gamma = g, kernel = 'rbf')
	clf.fit(x, y)
	plot_y.append(1 - clf.score(xt,yt))

plot_x, plot_y = np.asarray(plot_x), np.asarray(plot_y)
plt.plot(plot_x, plot_y, marker = 'o', mec = 'red', mfc = 'red', linestyle = '--')
plt.xlabel(r'$log_{10} Gamma$')
plt.ylabel(r'$Eout$')
plt.savefig('15.png')
plt.show()
