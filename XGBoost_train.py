import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
import Fed_XGBboost
from Fed_XGBboost import FED_XGB
import Data
import numpy
import csv


def client_train(File_Name,gi,hi,round):
    df = pd.read_csv(File_Name)
    X_train = df[df.columns[:-1].tolist()]
    y_train = df[df.columns[-1]]
    FML = FED_XGB(learning_rate=0.1,
                  n_estimators=1  # 总共迭代次数，每进行一轮进行一次全局更新
                  , max_depth=4, min_child_weight=0.2, gamma=0.03, )
    # 得到y_hat
    client_y_hat = FML.fit(X_train, y_train,gi,hi,round)
    # 返回一阶导和二阶导
    return [FML._grad(client_y_hat,y_train),FML._hess(client_y_hat,y_train)]
    # FML.save_model('model/clFML-mode.model')
