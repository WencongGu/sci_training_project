import numpy as np
import pandas as pd
import sympy
import time
from numba import jit

# class FED_XGB:
#     def __init__(self, base_score=0.5, max_depth=3, n_estimators=10, learning_rate=0.1, reg_lambda=1.,
#                  gamma=0., min_child_sample=10, min_child_weight=0.001, objective='linear'):
#         self.base_score = base_score  # 最开始时给叶子节点权重所赋的值，默认0.5，迭代次数够多的话，结果对这个初值不敏感
#         self.max_depth = max_depth  # 最大数深度，3
#         self.n_estimators = n_estimators  # 树的个数，10
#         self.rate = learning_rate  # 学习率，别和梯度下降里的学习率搞混了，这里是每棵树要乘以的权重系数，0.1
#         self.reg_lambda = reg_lambda  # L2正则项的权重系数
#         self.gamma = gamma  # 正则项中，叶子节点数T的权重系数
#         self.min_child_sample = min_child_sample  # 每个叶子节点的样本数（自己加的）
#         self.min_child_weight = min_child_weight  # 每个叶子节点的Hessian矩阵和，下面代码会细讲
#         self.objective = objective  # 目标函数
#         self.tree_structure = {}  # 用一个字典来存储每一颗树的树结构
#         self.epsilon = 0.00001  # 树生长最小增益
#         self.loss = lambda y, y_hat: (y - y_hat) ** 2. / 2.  # 损失函数
#         self.dependence = 0.3  # 客户端依赖系数（自己发明的）
#         self.epsilon0 = 0.994  # 主成分分析贡献率
#         self.m = 29  # 训练集特征个数
#         self.k = 0  # 主成分个数
#         self.mat = None  # 主成分分析中线性变换的矩阵的前k列
#
#     def xgb_cart_tree(self, X, w, depth=0):
#         """
#         递归构造XCart树
#         """
#         if depth > self.max_depth:
#             return None
#         best_var, best_cut = None, None
#         max_gain = 0.  # 这里增益的初值一定要设置为0，相当于对树做剪枝，即如果算出的增益小于0则不做分裂
#         G_left_best, G_right_best, H_left_best, H_right_best = 0., 0., 0., 0.
#         # 遍历每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
#         for item in range(self.k):
#             for cut in X[item]:
#                 if self.min_child_sample:  # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
#                     if (X.loc[X[item] < cut].shape[0] < self.min_child_sample) \
#                             | (X.loc[X[item] >= cut].shape[0] < self.min_child_sample):
#                         continue
#                 G_left = X.loc[X[item] < cut, 'g'].sum()
#                 G_right = X.loc[X[item] >= cut, 'g'].sum()
#                 H_left = X.loc[X[item] < cut, 'h'].sum()
#                 H_right = X.loc[X[item] >= cut, 'h'].sum()
#                 """!待完善：根据损失函数计算"""
#                 # min_child_weight指每个叶子节点上的H，即目标函数二阶导的加和
#                 if self.min_child_weight and ((H_left < self.min_child_weight) or (H_right < self.min_child_weight)):
#                     continue
#                 gain = G_left ** 2. / (H_left + self.reg_lambda) + G_right ** 2. / (H_right + self.reg_lambda)
#                 gain = gain - (G_left + G_right) ** 2. / (H_left + H_right + self.reg_lambda)
#                 gain = gain / 2. - self.gamma
#                 if gain > max_gain:
#                     best_var, best_cut = item, cut
#                     max_gain = gain
#                     G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
#         # 如果遍历完找不到可分列的点，或达到最大深度，或最大增益小于指定值，则返回None。否则递归生成树结构的字典，保存在self.tree_structure中
#         if best_var is None or max_gain <= self.epsilon:
#             return None
#         else:
#             id_left = X.loc[X[best_var] < best_cut].index.tolist()
#             w_left = - G_left_best / (H_left_best + self.reg_lambda)
#             id_right = X.loc[X[best_var] >= best_cut].index.tolist()
#             w_right = - G_right_best / (H_right_best + self.reg_lambda)
#             w[id_left] = w_left
#             w[id_right] = w_right
#             tree = {(best_var, best_cut): {}}
#             tree[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree(X.loc[id_left], w, depth + 1)
#             tree[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree(X.loc[id_right], w, depth + 1)
#         return tree
#
#     def _grad(self, y_hat, Y):
#         """
#         计算目标函数的一阶导
#         支持linear和logistic
#         """
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat - Y
#         elif self.objective == 'linear':
#             return y_hat - Y
#         else:
#             raise KeyError('temporarily: use linear or logistic')
#
#     def _hess(self, y_hat, Y):
#         """
#         计算目标函数的二阶导
#         支持linear和logistic
#         """
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat * (1.0 - y_hat)
#         elif self.objective == 'linear':
#             return np.array([1.] * Y.shape[0])
#         else:
#             raise KeyError('temporarily: use linear or logistic')
#
#     def fit(self, X, Y, gi, hi, r):
#         """
#         根据训练数据集X和标签集Y训练出树结构和权重
#         """
#         if X.shape[0] != Y.shape[0]:
#             raise ValueError('X and Y must have the same length!')
#         X = X.reset_index(drop=True)
#         self.pca_mat(X)
#         Z = self.pca(X)
#         Y = Y.values
#         # 这里根据base_score参数设定权重初始值
#         y_hat = np.array([self.base_score] * Y.shape[0])
#         for t in range(self.n_estimators):
#             print(f'fitting tree {t + 1}.')
#             print('原始数据规模',X.shape)
#             print('降维后规模',Z.shape)
#             if r == 0:
#                 Z['g'] = self._grad(y_hat, Y)
#                 Z['h'] = self._hess(y_hat, Y)
#             else:
#                 Z['g'] = self._grad(y_hat, Y)
#                 Z['h'] = self._hess(y_hat, Y)
#                 for i in range(int(self.dependence * min(len(Z['g']), len(gi)))):
#                     Z['g'][i] = gi[i]
#                     Z['h'][i] = hi[i]
#
#             f_t = pd.Series([0.] * Y.shape[0])
#             self.tree_structure[t + 1] = self.xgb_cart_tree(Z, f_t)
#             y_hat = y_hat + self.rate * f_t
#             print(f'tree {t + 1} fitted.')
#         print('All fitted.')
#         print(self.tree_structure)
#         return [y_hat, Z['g'], Z['h']]
#
#     def _get_tree_node_w(self, X, tree, w):
#         """
#         递归解构树结构，更新w为节点值
#         """
#         if tree is not None:
#             k = list(tree.keys())[0]
#             var, cut = k[0], k[1]
#             X_left = X.loc[X[var] < cut]
#             id_left = X_left.index.tolist()
#             X_right = X.loc[X[var] >= cut]
#             id_right = X_right.index.tolist()
#             for kk in tree[k].keys():
#                 if kk[0] == 'left':
#                     tree_left = tree[k][kk]
#                     w[id_left] = kk[1]
#                 elif kk[0] == 'right':
#                     tree_right = tree[k][kk]
#                     w[id_right] = kk[1]
#             self._get_tree_node_w(X_left, tree_left, w)
#             self._get_tree_node_w(X_right, tree_right, w)
#         return
#
#     def predict_raw(self, X):
#         """
#         根据训练结果预测返回原始预测值。迭代后y_t为前n-1棵树的y_hat，加和后返回n棵树的y_hat，Y
#         """
#         X = X.reset_index(drop=True)
#         Z = self.pca(X)
#         Y = pd.Series([self.base_score] * Z.shape[0])
#         for t in range(self.n_estimators):
#             tree = self.tree_structure[t + 1]
#             y_t = pd.Series([0.] * Z.shape[0])
#             self._get_tree_node_w(Z, tree, y_t)
#             Y = Y + self.rate * y_t
#         return Y
#
#     def predict_prob(self, X: pd.DataFrame):
#         """
#         当指定objective为logistic时，输出概率要做一个logistic转换
#         """
#
#         Y = self.predict_raw(X)
#         Y = Y.apply(lambda x: 1 / (1 + np.exp(-x)))
#
#         return Y
#
#     def pca(self, data0):
#         """
#         主成分分析，数据降维，用以提高程序运行速度。
#         先使用训练数据集调用pca_mat函数生成正交矩阵，再进行数据降维。
#         :param data0: 数据集
#         :return: 原数据变为k个主成分列，k个主成分列命名为数字0~k-1，它们的方差贡献率为self.epsilon0
#         """
#         data = data0.iloc[:, :self.m]
#         tran_data = data.T
#         br = np.array(tran_data)
#         for i in range(self.k):
#             data[i] = list(np.dot(self.mat[i, :], br))
#         data = data.loc[:, list(range(self.k))]
#         return data
#
#     def pca_mat(self, data0):
#         data = data0.iloc[:, :self.m]
#         tran_data = data.T
#         ar = np.array(data)
#         br = np.array(tran_data)
#         S = np.dot(br, ar)
#         S = S / (ar.shape[0] - 1)
#         eigval, eigvect = np.linalg.eig(S)
#         p = []
#         for i in range(eigvect.shape[0]):
#             p.append(list(eigvect[:, i]))
#             p[i].append(eigval[i])
#         p.sort(key=lambda x: x[len(eigvect)], reverse=True)
#         at = eigval.sum()
#         sum_ = 0
#         for i in range(len(eigval)):
#             sum_ += p[i][len(eigval)]
#             if sum_ >= self.epsilon0 * at:
#                 self.k = i + 1
#                 break
#         for i in range(eigvect.shape[0]):
#             p[i].pop()
#         self.mat = np.array(p[0:self.k])
#         return self.mat
#
#     def test(self, data):
#         import time
#         t1 = time.time()
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         y = data.iloc[:, -1]
#         y_hat = self.predict_raw(data)
#         plt.figure(figsize=(10, 20))
#         x = np.arange(len(y))
#         plt.plot(x, y, label='y')
#         plt.plot(x, y_hat, label='y_hat')
#         # plt.plot(x, y-y_hat,label='y-y_hat')
#         plt.title('accuracy line plot')
#         plt.legend()
#         plt.show()
#         plt.figure(figsize=(10, 20))
#         sns.swarmplot(y, y_hat)
#         plt.title('scatter plot')
#         plt.show()
#         t2 = time.time()
#         print('time', t2 - t1, ' s')
#         time.sleep(5)
# class FED_XGB:
#
#     def __init__(self,
#                  base_score=0.5,
#                  max_depth=3,
#                  n_estimators=10,
#                  learning_rate=0.1,
#                  reg_lambda=1,
#                  gamma=0,
#                  min_child_sample=None,
#                  min_child_weight=1,
#                  objective='linear'):
#
#         self.base_score = base_score  # 最开始时给叶子节点权重所赋的值，默认0.5，迭代次数够多的话，结果对这个初值不敏感
#         self.max_depth = max_depth  # 最大数深度
#         self.n_estimators = n_estimators  # 树的个数
#         self.learning_rate = learning_rate  # 学习率，别和梯度下降里的学习率搞混了，这里是每棵树要乘以的权重系数
#         self.reg_lambda = reg_lambda  # L2正则项的权重系数
#         self.gamma = gamma  # 正则项中，叶子节点数T的权重系数
#         self.min_child_sample = min_child_sample  # 每个叶子节点的样本数（自己加的）
#         self.min_child_weight = min_child_weight  # 每个叶子节点的Hessian矩阵和，下面代码会细讲
#         self.objective = objective  # 目标函数，可选linear和logistic
#         self.tree_structure = {}  # 用一个字典来存储每一颗树的树结构
#         self.dependence = 0.3  # 客户端依赖系数（自己发明的）
#         self.epsilon0 = 0.85  # 主成分分析贡献率
#         self.m = 29  # 训练集特征个数
#         self.k = 0  # 主成分个数
#         self.mat = None  # 主成分分析中线性变换的矩阵的前k列
#
#     def xgb_cart_tree(self, X, w, m_dpth):
#         '''
#         递归的方式构造XGB中的Cart树
#         X：训练数据集
#         w：每个样本的权重值，递归赋值
#         m_dpth：树的深度
#         '''
#
#         # 边界条件：递归到指定最大深度后，跳出
#         if m_dpth > self.max_depth:
#             return
#
#         best_var, best_cut = None, None
#         # 这里增益的初值一定要设置为0，相当于对树做剪枝，即如果算出的增益小于0则不做分裂
#         max_gain = 0
#         G_left_best, G_right_best, H_left_best, H_right_best = 0, 0, 0, 0
#         # 遍历每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
#         for item in [x for x in X.columns if x not in ['g', 'h', 'y']]:
#             for cut in list(set(X[item])):
#
#                 # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
#                 if self.min_child_sample:
#                     if (X.loc[X[item] < cut].shape[0] < self.min_child_sample) \
#                             | (X.loc[X[item] >= cut].shape[0] < self.min_child_sample):
#                         continue
#
#                 G_left = X.loc[X[item] < cut, 'g'].sum()
#                 G_right = X.loc[X[item] >= cut, 'g'].sum()
#                 H_left = X.loc[X[item] < cut, 'h'].sum()
#                 H_right = X.loc[X[item] >= cut, 'h'].sum()
#
#                 # min_child_weight在这里起作用，指的是每个叶子节点上的H，即目标函数二阶导的加和
#                 # 当目标函数为linear，即1/2*(y-y_hat)**2时，它的二阶导是1，那min_child_weight就等价于min_child_sample
#                 # 当目标函数为logistic，其二阶导为sigmoid(y_hat)*(1-sigmoid(y_hat))，可理解为叶子节点的纯度
#                 if self.min_child_weight:
#                     if (H_left < self.min_child_weight) | (H_right < self.min_child_weight):
#                         continue
#
#                 gain = G_left ** 2 / (H_left + self.reg_lambda) + \
#                        G_right ** 2 / (H_right + self.reg_lambda) - \
#                        (G_left + G_right) ** 2 / (H_left + H_right + self.reg_lambda)
#                 gain = gain / 2 - self.gamma
#                 if gain > max_gain:
#                     best_var, best_cut = item, cut
#                     max_gain = gain
#                     G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
#
#         # 如果遍历完找不到可分列的点，则返回None
#         if best_var is None:
#             return None
#
#         # 给每个叶子节点上的样本分别赋上相应的权重值
#         id_left = X.loc[X[best_var] < best_cut].index.tolist()
#         w_left = - G_left_best / (H_left_best + self.reg_lambda)
#
#         id_right = X.loc[X[best_var] >= best_cut].index.tolist()
#         w_right = - G_right_best / (H_right_best + self.reg_lambda)
#
#         w[id_left] = w_left
#         w[id_right] = w_right
#
#         # 用俄罗斯套娃式的json串把树的结构给存下来
#         tree_structure = {(best_var, best_cut): {}}
#         tree_structure[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree(X.loc[id_left], w, m_dpth + 1)
#         tree_structure[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree(X.loc[id_right], w, m_dpth + 1)
#
#         return tree_structure
#
#     def _grad(self, y_hat, Y):
#         '''
#         计算目标函数的一阶导
#         支持linear和logistic
#         '''
#
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat - Y
#         elif self.objective == 'linear':
#             return y_hat - Y
#         else:
#             raise KeyError('objective must be linear or logistic!')
#
#     def _hess(self, y_hat, Y):
#         '''
#         计算目标函数的二阶导
#         支持linear和logistic
#         '''
#
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat * (1.0 - y_hat)
#         elif self.objective == 'linear':
#             return np.array([1] * Y.shape[0])
#         else:
#             raise KeyError('objective must be linear or logistic!')
#
#     def fit(self, X: pd.DataFrame, Y, gi, hi, round):
#         '''
#         根据训练数据集X和Y训练出树结构和权重
#         '''
#
#         if X.shape[0] != Y.shape[0]:
#             raise ValueError('X and Y must have the same length!')
#         X = X.reset_index(drop='True')
#         # self.pca_mat(X)
#         # X = self.pca(X)
#         Y = Y.values
#         # 这里根据base_score参数设定权重初始值
#         y_hat = np.array([self.base_score] * Y.shape[0])
#         for t in range(self.n_estimators):
#             print('fitting tree {}...'.format(t + 1))
#             if round == 0:
#                 X['g'] = self._grad(y_hat, Y)
#                 X['h'] = self._hess(y_hat, Y)
#             else:
#                 X['g'] = self._grad(y_hat, Y)
#                 X['h'] = self._hess(y_hat, Y)
#                 for i in range(int(self.dependence * min(len(X['g']), len(gi)))):
#                     X['g'][i] = gi[i]
#                     X['h'][i] = hi[i]
#
#             f_t = pd.Series([0] * Y.shape[0])
#             self.tree_structure[t + 1] = self.xgb_cart_tree(X, f_t, 2)
#
#             y_hat = y_hat + self.learning_rate * f_t
#
#             print('tree {} fit done!'.format(t + 1))
#
#         print(self.tree_structure)
#         return [y_hat, X['g'], X['h']]
#
#     def _get_tree_node_w(self, X, tree, w):
#         '''
#         以递归的方法，把树结构解构出来，把权重值赋到w上面
#         '''
#
#         if not tree is None:
#             k = list(tree.keys())[0]
#             var, cut = k[0], k[1]
#             X_left = X.loc[X[var] < cut]
#             id_left = X_left.index.tolist()
#             X_right = X.loc[X[var] >= cut]
#             id_right = X_right.index.tolist()
#             for kk in tree[k].keys():
#                 if kk[0] == 'left':
#                     tree_left = tree[k][kk]
#                     w[id_left] = kk[1]
#                 elif kk[0] == 'right':
#                     tree_right = tree[k][kk]
#                     w[id_right] = kk[1]
#
#             self._get_tree_node_w(X_left, tree_left, w)
#             self._get_tree_node_w(X_right, tree_right, w)
#
#     def predict_raw(self, X: pd.DataFrame):
#         '''
#         根据训练结果预测
#         返回原始预测值
#         '''
#
#         X = X.reset_index(drop='True')
#         # X = self.pca(X)
#         Y = pd.Series([self.base_score] * X.shape[0])
#
#         for t in range(self.n_estimators):
#             tree = self.tree_structure[t + 1]
#             y_t = pd.Series([0] * X.shape[0])
#             self._get_tree_node_w(X, tree, y_t)
#             Y = Y + self.learning_rate * y_t
#
#         return Y
#
#     def predict_prob(self, X: pd.DataFrame):
#         '''
#         当指定objective为logistic时，输出概率要做一个logistic转换
#         '''
#
#         Y = self.predict_raw(X)
#         sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         Y = Y.apply(sigmoid)
#
#         return Y
#
#     def pca(self, data):
#
#         """
#         主成分分析，用以提高程序运行速度。
#         :param data: 数据集
#         :return: 原数据变为k个主成分列，k个主成分列命名为数字0~k-1，它们的方差贡献率为self.epsilon0
#         """
#
#         # data = data.iloc[:, self.m]
#         train_data = data.transpose()
#         br = np.array(train_data)
#         for i in range(self.k):
#             data[i] = list(np.dot(self.mat[i, :], br))
#         data = data.loc[:, list(range(self.k))]
#         return data
#
#     def pca_mat(self, data):
#         data = data.iloc[:, :self.m]
#         tran_data = data.transpose()
#         ar = np.array(data)
#         br = np.array(tran_data)
#         S = np.dot(br, ar)
#         S = S / (ar.shape[0] - 1)
#         eigval, eigvect = np.linalg.eig(S)
#         p = []
#         for i in range(eigvect.shape[0]):
#             p.append(list(eigvect[i, :]))
#             p[i].append(eigval[i])
#         p.sort(key=lambda x: x[len(eigvect)], reverse=True)
#         at = eigval.sum()
#         sum_ = 0
#         for i in range(len(eigval)):
#             sum_ += p[i][len(eigval)]
#             if sum_ >= self.epsilon0 * at:
#                 self.k = i + 1
#                 break
#         for i in range(eigvect.shape[0]):
#             p[i].pop()
#         self.mat = np.array(p[0:self.k])
#         return self.mat


# class FED_XGB:
#     def __init__(self, base_score=0.5, max_depth=5, n_estimators=10, learning_rate=0.1, reg_lambda=1.,
#                  gamma=0.1, min_child_sample=10, min_child_weight=0.00001, objective='linear'):
#         self.base_score = base_score  # 最开始时给叶子节点权重所赋的值，默认0.5，迭代次数够多的话，结果对这个初值不敏感
#         self.max_depth = max_depth  # 最大数深度，3
#         self.n_estimators = n_estimators  # 树的个数，10
#         self.rate = learning_rate  # 学习率，别和梯度下降里的学习率搞混了，这里是每棵树要乘以的权重系数，0.1
#         self.reg_lambda = reg_lambda  # L2正则项的权重系数
#         self.gamma = gamma  # 正则项中，叶子节点数T的权重系数
#         self.min_child_sample = min_child_sample  # 每个叶子节点的样本数（自己加的）
#         self.min_child_weight = min_child_weight  # 每个叶子节点的Hessian矩阵和，下面代码会细讲
#         self.objective = objective  # 目标函数
#         self.tree_structure = {}  # 用一个字典来存储每一颗树的树结构
#         self.epsilon = 0.00001  # 树生长最小增益
#         self.loss = lambda y, y_hat: (y - y_hat) ** 2. / 2.  # 损失函数
#
#     def xgb_cart_tree(self, X, w, dep=0):
#         """
#         递归构造XCart树
#         @param X:
#         @param w:
#         @param dep:
#         @return:
#         """
#         X = X.reset_index(drop='True')
#         best_var, best_cut = None, None
#         max_gain = 0.  # 这里增益的初值一定要设置为0，相当于对树做剪枝，即如果算出的增益小于0则不做分裂
#         G_left_best, G_right_best, H_left_best, H_right_best = 0., 0., 0., 0.
#         # 遍历每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
#         for item in [x for x in X.columns if x not in ['g', 'h', 'y', 'Amount', 'Class']]:
#             for cut in X[item]:
#                 if self.min_child_sample:  # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
#                     if (X.loc[X[item] < cut].shape[0] < self.min_child_sample) \
#                             | (X.loc[X[item] >= cut].shape[0] < self.min_child_sample):
#                         continue
#                 list_L = X.loc[X[item] < cut].index.tolist()
#                 list_R = X.loc[X[item] >= cut].index.tolist()
#                 G_left = X.loc[list_L, 'g'].sum()
#                 G_right = X.loc[list_R, 'g'].sum()
#                 H_left = len(list_L)
#                 H_right = len(list_R)
#                 """!待完善：根据损失函数计算"""
#                 # min_child_weight指每个叶子节点上的H，即目标函数二阶导的加和
#                 # if self.min_child_weight and ((H_left < self.min_child_weight) or (H_right < self.min_child_weight)):
#                 #     continue
#                 gain = G_left ** 2. / (H_left + self.reg_lambda) + G_right ** 2. / (H_right + self.reg_lambda)
#                 gain = gain - (G_left + G_right) ** 2. / (H_left + H_right + self.reg_lambda)
#                 gain = gain / 2. - self.gamma
#                 if gain > max_gain:
#                     best_var, best_cut = item, cut
#                     max_gain = gain
#                     G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
#         # 如果遍历完找不到可分列的点，或达到最大深度，或最大增益小于指定值，则返回None。否则递归生成树结构的字典，保存在self.tree_structure中
#         if best_var is None or dep > self.max_depth or max_gain <= self.epsilon:
#             return None
#         else:
#             id_left = X.loc[X[best_var] < best_cut].index.tolist()
#             w_left = - G_left_best / (H_left_best + self.reg_lambda)
#             id_right = X.loc[X[best_var] >= best_cut].index.tolist()
#             w_right = - G_right_best / (H_right_best + self.reg_lambda)
#             w[id_left] = w_left
#             w[id_right] = w_right
#             tree = {(best_var, best_cut): {}}
#             tree[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree(X.loc[id_left], w, dep + 1)
#             tree[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree(X.loc[id_right], w, dep + 1)
#         return tree
#
#     def fit(self, X, Y, gi, hi, r):
#         """
#         根据训练数据集X和标签集Y训练出树结构和权重
#         :param X:训练数据集X
#         :param Y:标签集
#         :param gi:首轮更新一阶导函数
#         :param hi:首轮二阶导
#         :param r:轮
#         :return:树结构字典
#         """
#         if X.shape[0] != Y.shape[0]:
#             raise ValueError('X and Y must have the same length!')
#         X = X.reset_index(drop=True)
#         Y = Y.values
#         # 这里根据base_score参数设定权重初始值
#         y_hat = pd.Series([self.base_score] * Y.shape[0])
#         for t in range(self.n_estimators):
#             print(f'fitting tree {t + 1}.')
#             if r == 0:
#                 X['g'] = self.diff(y_hat, Y)[0]
#                 X['h'] = self.diff(y_hat, Y)[1]
#             else:
#                 X['g'] = gi
#                 X['h'] = hi
#             f_t = pd.Series([0.] * Y.shape[0])
#             self.tree_structure[t] = self.xgb_cart_tree(X, f_t)
#             y_hat = y_hat + self.rate * f_t
#             # python中函数传参时传的是引用，所以这里f_t作为参数调用了xbg_cart_tree后其本身也得到更新
#             print(f'tree {t + 1} fitted.')
#         print('All fitted.')
#         return y_hat
#
#     def _get_tree_node_w(self, X, tree, w):
#         """
#         递归解构树结构，更新w为节点值
#         @param X:
#         @param tree:
#         @param w:
#         @return:
#         """
#         if tree is not None:
#             k = list(tree.keys())[0]
#             var, cut = k[0], k[1]
#             X_left = X.loc[X[var] < cut]
#             id_left = X_left.index.tolist()
#             X_right = X.loc[X[var] >= cut]
#             id_right = X_right.index.tolist()
#             '''
#             for kk in tree[k].keys():
#                 if kk[0] == 'left':
#                     tree_left = tree[k][kk]
#                     w[id_left] = kk[1]
#                 elif kk[0] == 'right':
#                     tree_right = tree[k][kk]
#                     w[id_right] = kk[1]
#                 和下面的代码等价，节省计算资源
#             '''
#             kk = list(tree[k].keys())
#             tree_left = tree[k][kk[0]]
#             w[id_left] = kk[0][1]
#             tree_right = tree[k][kk[1]]
#             w[id_right] = kk[1][1]
#             self._get_tree_node_w(X_left, tree_left, w)
#             self._get_tree_node_w(X_right, tree_right, w)
#         return
#
#     def predict_raw(self, X):
#         """
#         根据训练结果预测返回原始预测值。迭代后y_t为前n-1棵树的y_hat，加和后返回n棵树的y_hat，Y
#         @param X:
#         @return:
#         """
#         X = X.reset_index(drop=True)
#         Y = pd.Series([self.base_score] * X.shape[0])
#         for t in range(self.n_estimators):
#             tree = self.tree_structure[t]
#             y_t = pd.Series([0.] * X.shape[0])
#             self._get_tree_node_w(X, tree, y_t)
#             Y = Y + self.rate * y_t
#         return Y
#
#     def _grad(self, y_hat, Y):
#         """
#         计算目标函数的一阶导
#         支持linear和logistic
#         @param y_hat:
#         @param Y:
#         @return:
#         """
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat - Y
#         elif self.objective == 'linear':
#             return y_hat - Y
#         else:
#             raise KeyError('temporarily: use linear or logistic')
#
#     def _hess(self, y_hat, Y):
#         """
#         计算目标函数的二阶导，支持linear和logistic
#         @param y_hat:
#         @param Y:
#         @return:
#         """
#         if self.objective == 'logistic':
#             y_hat = 1.0 / (1.0 + np.exp(-y_hat))
#             return y_hat * (1.0 - y_hat)
#         elif self.objective == 'linear':
#             return np.array([1.] * Y.shape[0])
#         else:
#             raise KeyError('temporarily: use linear or logistic')
#
#     def find_G_H(self, y, y_hat, list_L_R):
#         """
#         求G、H
#         :param y:标签值
#         :param y_hat:预测值
#         :param list_L_R:索引列表
#         :return:求得G、H
#         """
#         if self.loss == 'linear':
#             G = y_hat[list_L_R].sum() - y[list_L_R].sum()
#             H = len(list_L_R)
#         elif self.loss == 'logistic':
#             pass  # 后续补充
#         else:
#             pass
#         return G, H
#
#     def diff(self, y_hat, y):
#         """
#         求导。linear和logistic直接计算，节省时间
#         :param y: y
#         :param y_hat:y_hat
#         :return: 导函数的符号表达式
#         """
#         m = sympy.Symbol('m')
#         n = sympy.Symbol('n')
#         fun = self.loss(m, n)
#         d1f = sympy.diff(fun, 'n')
#         d2f = sympy.diff(fun, 'n', 2)
#         g = []
#         h = []
#         for i in range(len(y)):
#             g.append(d1f.subs(m, y[i]).subs(n, y_hat[i]))
#             h.append(d2f.subs(m, y[i]).subs(n, y_hat[i]))  # 可以优化，避免for循环直接对y和y_hat全部带入导数求值
#         return g, h
from xgboost import Booster


class FED_XGB:
    def __init__(self, base_score=0.5, max_depth=3, n_estimators=10, learning_rate=0.1, reg_lambda=-1,
                 gamma=0., min_child_sample=10, min_child_weight=0.001, objective='linear'):
        self.base_score = base_score  # 最开始时给叶子节点权重所赋的值，默认0.5，迭代次数够多的话，结果对这个初值不敏感
        self.max_depth = max_depth  # 最大数深度，3
        self.n_estimators = n_estimators  # 树的个数，10
        self.rate = learning_rate  # 学习率，别和梯度下降里的学习率搞混了，这里是每棵树要乘以的权重系数，0.1
        self.reg_lambda = reg_lambda  # L2正则项的权重系数
        self.gamma = gamma  # 正则项中，叶子节点数T的权重系数
        self.min_child_sample = min_child_sample  # 每个叶子节点的样本数（自己加的）
        self.min_child_weight = min_child_weight  # 每个叶子节点的Hessian矩阵和，下面代码会细讲
        self.objective = objective  # 目标函数
        self.tree_structure = {}  # 用一个字典来存储每一颗树的树结构
        self.epsilon = 0.00001  # 树生长最小增益
        self.loss = lambda y, y_hat: (y - y_hat) ** 2. / 2.  # 损失函数
        self.dependence = 0.3  # 客户端依赖系数（自己发明的）
        self.epsilon0 = 0.993  # 主成分分析贡献率
        self.m = 28  # 训练集特征个数
        self.k = 0  # 主成分个数
        self.mat = None  # 主成分分析中线性变换的矩阵的前k列

    # @jit
    def xgb_cart_tree(self, X, w, depth=0):
        """
        递归构造XCart树
        """
        if depth > self.max_depth:
            return None
        best_var, best_cut = None, None
        max_gain = 0
        G_left_best, G_right_best, H_left_best, H_right_best = 0, 0, 0, 0
        for item in [x for x in X.columns if x not in ['g', 'h', 'Class']]:
            for cut in X[item].drop_duplicates():  # 遍历每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
                if self.min_child_sample:  # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
                    if (X.loc[X[item] < cut].shape[0] < self.min_child_sample) \
                            | (X.loc[X[item] >= cut].shape[0] < self.min_child_sample):
                        continue
                G_left = X.loc[X[item] < cut, 'g'].sum()
                G_right = X.loc[X[item] >= cut, 'g'].sum()
                H_left = X.loc[X[item] < cut, 'h'].sum()
                H_right = X.loc[X[item] >= cut, 'h'].sum()
                if self.min_child_weight:
                    if (H_left < self.min_child_weight) | (H_right < self.min_child_weight):
                        continue
                gain = G_left ** 2 / (H_left + self.reg_lambda) + G_right ** 2 / (H_right + self.reg_lambda)
                gain = gain - (G_left + G_right) ** 2 / (H_left + H_right + self.reg_lambda)
                gain = gain / 2 - self.gamma
                if gain > max_gain:
                    best_var, best_cut = item, cut
                    max_gain = gain
                    G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
        if best_var is None or max_gain <= self.epsilon:
            return None
        else:
            id_left = X.loc[X[best_var] < best_cut].index.tolist()
            w_left = - G_left_best / (H_left_best + self.reg_lambda)
            id_right = X.loc[X[best_var] >= best_cut].index.tolist()
            w_right = - G_right_best / (H_right_best + self.reg_lambda)
            w[id_left] = w_left
            w[id_right] = w_right
            tree = {(best_var, best_cut): {}}
            tree[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree(X.loc[id_left], w, depth + 1)  # 递归左子树
            tree[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree(X.loc[id_right], w, depth + 1)  # 递归右子树
        return tree

    # @jit
    def xgb_cart_tree_server(self, X, w, depth=0):
        """
        递归构造XCart树（并行化版本）
        """
        if depth > self.max_depth:
            return None
        best_var, best_cut = None, None
        max_gain = 0
        G_left_best, G_right_best, H_left_best, H_right_best = 0, 0, 0, 0
        client0 = X[0]
        for item in [x for x in client0.columns if x not in ['g', 'h', 'Class']]:
            for client in X:
                for cut in client[item].drop_duplicates():  # 遍历客户端每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
                    G_left = 0
                    G_right = 0
                    H_left = 0
                    H_right = 0
                    for client_j in X:  # 遍历每一个客户端
                        if self.min_child_sample:  # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
                            if (client_j.loc[client_j[item] < cut].shape[0] < self.min_child_sample) \
                                    | (client_j.loc[client_j[item] >= cut].shape[0] < self.min_child_sample):
                                continue
                        G_left += client_j.loc[client_j[item] < cut, 'g'].sum()
                        G_right += client_j.loc[client_j[item] >= cut, 'g'].sum()
                        H_left += client_j.loc[client_j[item] < cut, 'h'].sum()
                        H_right += client_j.loc[client_j[item] >= cut, 'h'].sum()
                    if self.min_child_weight:
                        if (H_left < self.min_child_weight) | (H_right < self.min_child_weight):
                            continue
                    gain = G_left ** 2 / (H_left + self.reg_lambda) + G_right ** 2 / (H_right + self.reg_lambda)
                    gain = gain - (G_left + G_right) ** 2 / (H_left + H_right + self.reg_lambda)
                    gain = gain / 2 - self.gamma
                    # gain = 0 - gain
                    if gain > max_gain:
                        best_var, best_cut = item, cut
                        max_gain = gain
                        G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
        if best_var is None or max_gain <= self.epsilon:
            return None
        else:
            # 给每个叶子节点上的样本分别赋上相应的权重值
            w_left = - G_left_best / (H_left_best + 1)
            w_right = - G_right_best / (H_right_best + 1)
            all_client_left = []
            for client in X:
                id_left = client.loc[client[best_var] < best_cut].index.tolist()
                w[id_left] = w_left
                all_client_left.append(client.loc[id_left])

            all_client_right = []
            for client in X:
                id_right = client.loc[client[best_var] >= best_cut].index.tolist()
                w[id_right] = w_right
                all_client_right.append(client.loc[id_right])
            tree = {(best_var, best_cut): {}}
            tree[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree_server(all_client_left, w,
                                                                                     depth + 1)  # 递归左子树
            tree[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree_server(all_client_right, w,
                                                                                       depth + 1)  # 递归右子树
        return tree

    # @jit
    def _grad(self, y_hat, Y):
        """
        计算目标函数的一阶导
        支持linear和logistic
        """
        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat - Y
        elif self.objective == 'linear':
            return y_hat - Y
        else:
            raise KeyError('temporarily: use linear or logistic')

    @jit
    def _hess(self, y_hat, Y):
        """
        计算目标函数的二阶导
        支持linear和logistic
        """
        if self.objective == 'logistic':
            y_hat = (1 - 1.0 / (1.0 + np.exp(-y_hat))) / (1. + np.exp(-y_hat))
            return y_hat * (1.0 - y_hat)
        elif self.objective == 'linear':
            return np.array([1.] * Y.shape[0])
        else:
            raise KeyError('temporarily: use linear or logistic')

    # @jit
    def fit_server(self, X, Y):
        """
        根据训练数据集X和标签集Y训练出树结构和权重(并行化版本)
        """
        # if X.shape[0] != Y.shape[0]:
        #     raise ValueError('X and Y must have the same length!')
        # X = X.reset_index(drop=True)
        df_y = None
        for client_y in Y:
            df_y = pd.concat([df_y, client_y], axis=0)
        Y = df_y.values
        y_hat = np.array([self.base_score] * Y.shape[0])
        t0 = time.time()
        for t in range(self.n_estimators):
            t1 = time.time()
            print(f'fitting tree {t + 1}.')
            i = 1
            index = 0  # 用于测试及Y对每个client的分片偏移量记录
            gi = self._grad(y_hat, Y)
            hi = self._hess(y_hat, Y)
            for client in X:
                # shape += client.shape()
                print(f'客户端 {i} 原始数据总规模', client.shape)

                # assert type(gi) == np.ndarray
                # assert type(hi) == np.ndarray
                client['g'] = gi[index:index + client.shape[0]]
                client['h'] = hi[index:index + client.shape[0]]
                index = index + client.shape[0]
                i += 1
            # if r == 0:
            # X['g'] = self._grad(y_hat, Y)
            # X['h'] = self._hess(y_hat, Y)
            # else:
            # X['g'] = self._grad(y_hat, Y)
            # X['h'] = self._hess(y_hat, Y)
            # for i in range(int(self.dependence * min(len(X['g']), len(gi)))):
            #     X['g'][i] = gi[i]
            #     X['h'][i] = hi[i]
            f_t = pd.Series([0.] * Y.shape[0])
            self.tree_structure[t + 1] = self.xgb_cart_tree_server(X, f_t)
            y_hat = y_hat + self.rate * f_t
            t2 = time.time()
            print(f'tree {t + 1} fitted. Time: {t2 - t1} s')
        tt = time.time()
        print(f'All fitted. Time: {tt - t0} s')
        print(self.tree_structure)
        return self.tree_structure

    # @jit
    def fit(self, X, Y):
        """
        根据训练数据集X和标签集Y训练出树结构和权重
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same length!')
        X = X.reset_index(drop=True)
        Y = Y.values
        y_hat = np.array([self.base_score] * Y.shape[0])
        t0 = time.time()
        for t in range(self.n_estimators):
            t1 = time.time()
            print(f'fitting tree {t + 1}.')
            print('原始数据规模', X.shape)
            print('降维后规模', X.shape)
            # if r == 0:
            X['g'] = self._grad(y_hat, Y)
            X['h'] = self._hess(y_hat, Y)
            # else:
            # X['g'] = self._grad(y_hat, Y)
            # X['h'] = self._hess(y_hat, Y)
            # for i in range(int(self.dependence * min(len(X['g']), len(gi)))):
            #     X['g'][i] = gi[i]
            #     X['h'][i] = hi[i]
            f_t = pd.Series([0.] * Y.shape[0])
            self.tree_structure[t + 1] = self.xgb_cart_tree(X, f_t)
            y_hat = y_hat + self.rate * f_t
            t2 = time.time()
            print(f'tree {t + 1} fitted. Time: {t2 - t1} s')
        tt = time.time()
        print(f'All fitted. Time: {tt - t0} s')
        print(self.tree_structure)
        return [y_hat, X['g'], X['h']]

    # @jit
    def _get_tree_node_w(self, X, tree, w):
        """
        递归解构树结构，更新w为节点值
        """
        if tree is not None:
            k = list(tree.keys())[0]
            var, cut = k[0], k[1]
            X_left = X.loc[X[var] < cut]
            id_left = X_left.index.tolist()
            X_right = X.loc[X[var] >= cut]
            id_right = X_right.index.tolist()
            for kk in tree[k].keys():
                if kk[0] == 'left':
                    tree_left = tree[k][kk]
                    w[id_left] = kk[1]
                elif kk[0] == 'right':
                    tree_right = tree[k][kk]
                    w[id_right] = kk[1]
            self._get_tree_node_w(X_left, tree_left, w)
            self._get_tree_node_w(X_right, tree_right, w)
        return

    # @jit
    def predict_raw(self, X):
        """
        根据训练结果预测返回原始预测值。迭代后y_t为前n-1棵树的y_hat，加和后返回n棵树的y_hat，Y
        """
        X = X.reset_index(drop=True)
        Y = pd.Series([self.base_score] * X.shape[0])
        for t in range(self.n_estimators):
            tree = self.tree_structure[t + 1]
            y_t = pd.Series([0.] * X.shape[0])
            self._get_tree_node_w(X, tree, y_t)
            Y = Y + self.rate * y_t
        return Y

    # @jit
    def predict_prob(self, X: pd.DataFrame):
        """
        当指定objective为logistic时，输出概率要做一个logistic转换
        """
        Y = self.predict_raw(X)
        # Y = Y.apply(lambda x: 1 / (1 + np.exp(-x)))
        return np.array(Y > 0.5, dtype='int')

    def pca(self, data0):
        """
        主成分分析，数据降维，用以提高程序运行速度。
        先使用训练数据集调用pca_mat函数生成正交矩阵，再进行数据降维。
        """
        data = data0.iloc[:, :self.m]
        tran_data = data.T
        br = np.array(tran_data)
        for i in range(self.k):
            data[i] = list(np.dot(self.mat[i, :], br))
        data = data.loc[:, list(range(self.k))]
        return data

    def pca_mat(self, data0):
        data = data0.iloc[:, :self.m]
        tran_data = data.T
        ar = np.array(data)
        br = np.array(tran_data)
        S = np.dot(br, ar)
        S = S / (ar.shape[0] - 1)
        eigval, eigvect = np.linalg.eig(S)
        p = []
        for i in range(eigvect.shape[0]):
            p.append(list(eigvect[:, i]))
            p[i].append(eigval[i])
        p.sort(key=lambda x: x[len(eigvect)], reverse=True)
        at = eigval.sum()
        sum_ = 0
        for i in range(len(eigval)):
            sum_ += p[i][len(eigval)]
            if sum_ >= self.epsilon0 * at:
                self.k = i + 1
                break
        for i in range(eigvect.shape[0]):
            p[i].pop()
        self.mat = np.array(p[0:self.k])
        return self.mat

    def test(self, data):
        import matplotlib.pyplot as plt
        y = data.iloc[:, -1]
        y_hat = self.predict_raw(data)
        plt.figure(figsize=(10, 20))
        x = np.arange(len(y))
        plt.plot(x, y, label='y')
        plt.plot(x, y_hat, label='y_hat')
        # plt.plot(x, y-y_hat,label='y-y_hat')
        plt.title('accuracy line plot')
        plt.legend()
        plt.show()
        time.sleep(5)
