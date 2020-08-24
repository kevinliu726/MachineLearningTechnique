import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
# data pre-processing
train = []
test = []
G_array = [1e-1, 1e0, 1e1, 1e2, 1e3]

with open("features.train", 'r') as f:
	for line in f:
		train.append([float(i) for i in line.split()])

train = np.asarray(train)

x = []
for i in train:
	digit, inten, symm = i[0], i[1], i[2]
	if digit == 0:
		x.append([inten, symm, 1])
	else:
		x.append([inten, symm,-1])
x = np.asarray(x)

G_hist = [0]*5
from sklearn import svm
for i in range(100):
	acc = []
	for g in G_array:
		clf = svm.SVC(C = 0.1, gamma = g, kernel = 'rbf')
		np.random.shuffle(x)
		xt = x[1000:,:2]
		yt = x[1000:,2]
		clf.fit(xt, yt)
		acc.append(1 - clf.score(x[:1000,:2],x[:1000,2]))
	idx = np.argmin(acc)
	G_hist[idx] += 1

hist = []
for i, j in enumerate(G_hist):
    hist += [np.log10(G_array[i])]*j
plt.hist(hist, bins=[-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.5)
plt.savefig('16.png')
plt.show()
