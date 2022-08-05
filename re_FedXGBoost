import numpy as np
import pandas as pd
import time


class FED_XGB:
    def __init__(self, base_score=0.5, max_depth=3, n_estimators=10, learning_rate=0.1, reg_lambda=1.,
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
        self.m = 29  # 训练集特征个数
        self.k = 0  # 主成分个数
        self.mat = None  # 主成分分析中线性变换的矩阵的前k列

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
            tree[(best_var, best_cut)][('left', w_left)] = self.xgb_cart_tree(X.loc[id_left], w, depth + 1)
            tree[(best_var, best_cut)][('right', w_right)] = self.xgb_cart_tree(X.loc[id_right], w, depth + 1)
        return tree

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

    def _hess(self, y_hat, Y):
        """
        计算目标函数的二阶导
        支持linear和logistic
        """
        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat * (1.0 - y_hat)
        elif self.objective == 'linear':
            return np.array([1.] * Y.shape[0])
        else:
            raise KeyError('temporarily: use linear or logistic')

    def fit(self, X, Y, gi, hi, r):
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
            if r == 0:
                X['g'] = self._grad(y_hat, Y)
                X['h'] = self._hess(y_hat, Y)
            else:
                X['g'] = self._grad(y_hat, Y)
                X['h'] = self._hess(y_hat, Y)
                for i in range(int(self.dependence * min(len(X['g']), len(gi)))):
                    X['g'][i] = gi[i]
                    X['h'][i] = hi[i]
            f_t = pd.Series([0.] * Y.shape[0])
            self.tree_structure[t + 1] = self.xgb_cart_tree(X, f_t)
            y_hat = y_hat + self.rate * f_t
            t2 = time.time()
            print(f'tree {t + 1} fitted. Time: {t2 - t1} s')
        tt = time.time()
        print(f'All fitted. Time: {tt - t0} s')
        print(self.tree_structure)
        return [y_hat, X['g'], X['h']]

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

    def predict_prob(self, X: pd.DataFrame):
        """
        当指定objective为logistic时，输出概率要做一个logistic转换
        """
        Y = self.predict_raw(X)
        Y = Y.apply(lambda x: 1 / (1 + np.exp(-x)))
        return Y

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
