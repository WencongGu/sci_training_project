import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
import Data
import numpy
import csv


def client_train(File_Name):
    df = pd.read_csv(File_Name)
    X_train = df[df.columns[:-1].tolist()]
    y_train = df[df.columns[-1]]
    FML = XGBClassifier(fl_split=405, tree_method='approx', updater='grow_histmaker,prune', learning_rate=0.1,
                        n_estimators=100  # 总共迭代次数，每进行一轮进行一次全局更新
                        , max_depth=4, min_child_weight=0.2, gamma=0.03, subsample=0.6, nthread=4,
                        scale_pos_weight=1, seed=27,
                        objective='reg:squaredlogerror')
    FML.fit(X_train, y_train)
    # FML.save_model('model/clFML-mode.model')
    return FML.booster().get_score(importance_type='gain')
