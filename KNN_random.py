# source
# https://blog.csdn.net/pengjunlee/article/details/82713047?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=1328592.7810.16147464119001617&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control

# KNN最邻近分类算法的实现原理：为了判断未知样本的类别，以所有已知类别的样本作为参照，计算未知样本与所有已知样本的距离，
# 从中选取与未知样本距离最近的K个已知样本，根据少数服从多数的投票法则（majority-voting），将未知样本与K个最邻近样本
# 中所属类别占比较多的归为一类。

import matplotlib.pyplot as plt
# 导入数组工具
import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成样本数为500，分类数为5的数据集
data = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, random_state=8)
X, Y = data

# 将生成的数据集进行可视化
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y,  cmap=plt.cm.spring, edgecolors='k')
plt.show()

clf = KNeighborsClassifier()
clf.fit(X, Y)

# 绘制图形
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")

# 把待分类的数据点用五星表示出来
plt.scatter(0, 5, marker='*', c='red', s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[0, 5]])
print(res)
plt.text(0.2, 4.6, 'Classification flag: ' + str(res))
plt.text(3.75, -13, 'Model accuracy: {:.2f}'.format(clf.score(X, Y)))

plt.show()
