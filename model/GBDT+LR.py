"""
LR、GBDT、GBDT+LR：
训练三个模型对数据进行预测， 分别是LR模型、GBDT模型和两者的组合模型（GBDT负责对各个特征进行交叉， 把原始特征向量转换为新的离散型特征向量， 
然后再使用逻辑回归模型）， 然后分别观察它们的预测效果。

对于不同的模型， 特征会有不同的处理方式，具体如下：

- 逻辑回归模型：连续特征归一化， 离散特征one-hot编码
- GBDT模型：离散特征one-hot编码（树模型连续特征不需要归一化）
- GBDT+LR模型：离散特征one-hot编码（由于LR使用的特征是GBDT的输出， 原数据依然是GBDT进行处理， 所以只需要离散特征one-hot编码）


任务：
开发预测广告点击率(CTR)的模型。即给定一个用户和他正在访问的页面，预测他点击给定广告的概率。


数据集：
Label：目标变量，0表示未点击， 1表示点击
l1-l13: 13列的数值特征，大部分是计数特征
C1-C26: 26列的分类特征，这些特征的值离散成了32位的数据
"""

# ==========================================================================================================

"""导包"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import gc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

"""数据读取"""

path = '../data/'
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')

"""数据预处理"""

# 去掉 Id 列
df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)

# 测试集新增 Label 列
df_test['Label'] = -1

# 合并测试集和训练集
data = pd.concat([df_train, df_test])

# 填充缺失值
data.fillna(-1, inplace=True)

"""划分连续和离散特征"""

continuous_fea = ['I'+str(i+1) for i in range(13)]
category_fea = ['C'+str(i+1) for i in range(26)]

# ==========================================================================================================

"""逻辑回归建模"""
def lr_model(data, category_fea, continuous_fea):
    """
        逻辑回归建模
        
        :param data: df, 数据表
        :param category_fea: list, 离散特征列
        :param continuous_fea: list, 连续特征列
    """
    
    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    
    # 离散特征one-hot编码
    for col in category_fea:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)
    
    # 拆分训练集和测试集
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)
    
    # 划分训练集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建立模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])  # −(ylog(p)+(1−y)log(1−p)) log_loss
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)
    
    # 模型预测
    y_pred = lr.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率，这里的1表示点击的概率
    print('predict: ', y_pred[:10]) # 这里看前10个，预测为点击的概率
    
"""训练和预测逻辑回归模型"""
lr_model(data.copy(), category_fea, continuous_fea)

# ==========================================================================================================

"""GBDT建模"""
def gbdt_model(data, category_fea, continuous_fea):
    """
        GBDT建模
        
        :param data: df, 数据表
        :param category_fea: list, 离散特征列
        :param continuous_fea: list, 连续特征列
    """
    
    # 离散特征one-hot编码
    for col in category_fea:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 拆分训练集和测试集
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)
    
    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)
    
    # 建立模型
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',  # 这里用gbdt
                             objective='binary', 
                             subsample=0.8,
                             min_child_weight=0.5, 
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=10000
                            )
    gbm.fit(x_train, y_train, 
            eval_set=[(x_train, y_train), (x_val, y_val)], 
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100,
           )
    
    tr_logloss = log_loss(y_train, gbm.predict_proba(x_train)[:, 1])   # −(ylog(p)+(1−y)log(1−p)) log_loss
    val_logloss = log_loss(y_val, gbm.predict_proba(x_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)
    
    # 模型预测
    y_pred = gbm.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率，这里的1表示点击的概率
    print('predict: ', y_pred[:10]) # 这里看前10个，预测为点击的概率
    
"""训练和预测GBDT模型"""
gbdt_model(data.copy(), category_fea, continuous_fea)

# ==========================================================================================================

"""GBDT+LR建模"""
def gbdt_lr_model(data, category_feature, continuous_feature):  # 0.43616
    """
        GBDT+LR建模
        
        :param data: df, 数据表
        :param category_fea: list, 离散特征列
        :param continuous_fea: list, 连续特征列
    """
    
    # 离散特征one-hot编码
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    # 拆分训练集和测试集
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis = 1, inplace = True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2020)

    # 建立GBDT模型
    gbm = lgb.LGBMClassifier(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.01,
                            n_estimators=1000,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            early_stopping_rounds = 100,
            )
    
    # GBDT负责对各个特征进行交叉， 把原始特征向量转换为新的离散型特征向量
    model = gbm.booster_

    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    # 新特征（连续特征 + 离散特征one-hot编码 + GBDT生成的离散特征）
    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])    
    del train
    del test
    gc.collect()  # 清理内存

    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    # GBDT生成的离散特征one-hot编码
    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    # 建立逻辑回归模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('tr-logloss: ', tr_logloss)
    print('val-logloss: ', val_logloss)
    
    # 模型预测
    y_pred = lr.predict_proba(test)[:, 1]
    print(y_pred[:10])

"""训练和预测GBDT+LR模型"""
gbdt_lr_model(data.copy(), category_fea, continuous_fea)
