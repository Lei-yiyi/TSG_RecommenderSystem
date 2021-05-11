"""
DeepFM：
DeepFM模型大致由两部分组成，一部分是FM，还有一部分就是DNN，而FM又由一阶特征部分与二阶特征交叉部分组成，所以可以将整个模型拆成三部分，
分别是FM的一阶特征，FM的二阶特征交叉以及DNN的高阶特征交叉。此外每一部分可能是由不同的特征组成，所以在构建模型的时候需要分别对这三部分
输入的特征进行选择。

linear_logits: 这一块主要是针对连续特征和离散特征，将连续特征和离散特征的onehot编码组成一维向量（实际应用中根据自己的业务放置不同
的一阶特征），最后计算FM的一阶特征（即FM的前半部分 𝑤1𝑥1 + 𝑤2𝑥2...𝑤𝑛𝑥𝑛 + 𝑏 的线性计算）的logits值

fm_logits: 这一块主要是针对离散特征，首先过embedding，然后使用FM特征交叉的方式（只考虑两两特征进行交叉）得到新的特征向量，最后计算
FM的二阶交叉特征的logits值

dnn_logits: 这一块主要是针对离散特征，首先过embedding，然后将得到的embedding拼接成一个向量，通过dnn学习特征之间的隐式特征交叉，
最后计算DNN的高阶交叉特征的logits值


任务：
开发预测广告点击率(CTR)的模型。即给定一个用户和他正在访问的页面，预测他点击给定广告的概率。


数据集：
Label：目标变量，0表示未点击， 1表示点击
l1 - l13: 13列的数值特征
C1 - C26: 26列的类别特征
"""


import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


"""数据预处理"""


def data_process(data_df, dense_features, sparse_features):
    # sparse特征填充缺失值
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    # 数值处理
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    # dense特征填充缺失值
    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    # 类别编码
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]


"""构建DeepFM模型——输入层"""


# 构建输入层，这里使用字典（dense和sparse两类字典）的形式返回，方便后续构建模型
def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)

    return dense_input_dict, sparse_input_dict


"""构建DeepFM模型——维度为k的embedding层"""


# 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()

    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    # 如果是用于线性部分的embedding层，其维度为1，否则维度就是自己定义的embedding维度
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)

    return embedding_layers_dict


"""构建DeepFM模型——输出层 linear_logits"""


# linear_logits计算dense特征的logits和sparse特征的logits两部分，最后计算FM的一阶特征的logits值
def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    """
        调用函数：build_embedding_layers
    """
    # 将所有的dense特征的Input层，然后经过一个全连接层得到dense特征的logits
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_inputs)

    # 获取linear部分sparse特征的embedding层，这里
    # 使用embedding层的原因是：对于linear部分直接将特征进行onehot然后通过一个全连接层，当维度特别大的时候，计算比较慢
    # 使用embedding层的好处是：可以通过查表的方式获取到哪些非零的元素对应的权重，然后在将这些权重相加，效率比较高
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)

    # 将一维的embedding拼接，注意这里需要使用一个Flatten层，使维度对应
    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))  # B x 1
        sparse_1d_embed.append(embed)

    # embedding中查询得到的权重就是对应onehot向量中一个位置的权重，所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 只不过是这里的输入特征只有0和1，所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)
    sparse_logits_output = Add()(sparse_1d_embed)

    # 最终将dense特征和sparse特征对应的logits相加，得到最终linear的logits
    linear_logits = Add()([dense_logits_output, sparse_logits_output])
    return linear_logits


"""构建DeepFM模型——输出层 fm_logits"""


class FM_Layer(Layer):
    def __init__(self):
        super(FM_Layer, self).__init__()

    def call(self, inputs):
        # 优化后的公式为： 0.5 * 求和（和的平方-平方的和）  =>> B x 1
        concated_embeds_value = inputs  # B x n x k

        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))  # B x 1 x k
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)  # B x1 xk
        cross_term = square_of_sum - sum_of_square  # B x 1 x k
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)  # B x 1

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


# fm_logits只考虑sparse特征的二阶交叉，将所有的embedding拼接到一起，最后计算FM的二阶交叉特征的logits值
def get_fm_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    """
        调用函数：FM_Layer
    """
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))

    # 只考虑sparse特征的二阶交叉，将所有的embedding拼接到一起，最后计算FM的二阶交叉特征的logits值
    # 因为类别型数据输入的只有0和1，所以不需要考虑将隐向量与x相乘，直接对隐向量进行操作即可
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)  # B x 1 x k
        sparse_kd_embed.append(_embed)

    # 将所有sparse的embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，k为embedding大小
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)  # B x n x k
    fm_cross_out = FM_Layer()(concat_sparse_kd_embed)

    return fm_cross_out


"""构建DeepFM模型——输出层 dnn_logits"""


# dnn_logits只考虑sparse特征的高阶交叉，将所有的embedding拼接到一起输入到dnn中，最后计算DNN的高阶交叉特征的logits值
def get_dnn_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))

    # 将所有非零的sparse特征对应的embedding拼接到一起
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)  # B x 1 x k
        _embed = Flatten()(_embed)  # B x k
        sparse_kd_embed.append(_embed)

    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)  # B x nk

    # dnn层，这里的Dropout参数，Dense中的参数都可以自己设定，以及Dense的层数都可以自行设定
    mlp_out = Dropout(0.5)(Dense(256, activation='relu')(concat_sparse_kd_embed))
    mlp_out = Dropout(0.3)(Dense(256, activation='relu')(mlp_out))
    mlp_out = Dropout(0.1)(Dense(256, activation='relu')(mlp_out))

    dnn_out = Dense(1)(mlp_out)

    return dnn_out


"""构建DeepFM模型——DeepFM模型"""


def DeepFM(linear_feature_columns, dnn_feature_columns):
    """
        调用函数：build_input_layers, build_embedding_layers, get_linear_logits, get_fm_logits, get_dnn_logits
    """

    """输入层"""
    # 构建输入层，这里使用字典（dense和sparse两类字典）的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # 构建输入层，输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    """维度为k的embedding层"""
    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    """输出层——linear_logits"""
    # 将linear部分的特征中sparse特征筛选出来，后面用来做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # linear_logits计算dense特征的logits和sparse特征的logits两部分，最后计算FM的一阶特征的logits值
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    """输出层——fm_logits"""
    # 将dnn部分的特征中sparse特征筛选出来
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # fm_logits只考虑sparse特征的二阶交叉，将所有的embedding拼接到一起，最后计算FM的二阶交叉特征的logits值
    fm_logits = get_fm_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    """输出层——dnn_logits"""
    # dnn_logits只考虑sparse特征的高阶交叉，将所有的embedding拼接到一起输入到dnn中，最后计算DNN的高阶交叉特征的logits值
    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    """输出层——logits（将linear_logits, fm_logits, dnn_logits相加作为最终的logits）"""
    output_logits = Add()([linear_logits, fm_logits, dnn_logits])

    """输出层——sigmoid（这里的激活函数使用sigmoid）"""
    output_layers = Activation("sigmoid")(output_logits)

    """模型"""
    model = Model(input_layers, output_layers)

    return model


if __name__ == "__main__":
    """读取数据"""
    path = '../data/'
    data_df = pd.read_csv(path + 'DeepFM_data.txt')

    """划分dense和sparse特征"""
    columns = data_df.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    """数据预处理"""
    train_data = data_process(data_df, dense_features, sparse_features)
    train_data['label'] = data_df['label']

    """划分 linear 和 dnn 特征(根据实际场景进行选择)，并将分组后的特征标记为 DenseFeat 和 SparseFeat"""
    # 使用具名元组定义特征标记
    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])

    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data_df[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                            for feat in dense_features]
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data_df[feat].nunique(), embedding_dim=4)
                           for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                         for feat in dense_features]

    """构建DeepFM模型"""
    history = DeepFM(linear_feature_columns, dnn_feature_columns)
    history.summary()
    history.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    """模型训练"""
    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data_df[name] for name in dense_features + sparse_features}

    history.fit(train_model_input, train_data['label'].values, batch_size=64, epochs=5, validation_split=0.2, )
