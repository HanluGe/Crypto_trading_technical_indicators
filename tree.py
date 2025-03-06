# tree.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # 进度条库

def get_features(df):
    """
    从 DataFrame 中提取 124 个技术指标，假设这些指标位于第 13 列到第 136 列（0 索引下为 12 到 135），
    并将其转换为 float64 类型的 NumPy 数组。
    """
    return df.iloc[:, 12:136].values.astype(np.float64)

def train_ensemble(train_df):
    """
    接受训练数据 DataFrame，提取 124 个因子数据（假设位于第 13 列到第 136 列）和标签，
    构建 100 棵决策树，并返回组成的树列表。
    在训练过程中显示进度条。
    """
    X = get_features(train_df)
    y = train_df['label'].astype(int)  # 将标签转换为整数

    trees = []
    for i in tqdm(range(100), desc="Training trees"):
        # 设置不同的 random_state 保证每棵树随机选择的分裂特征不同
        clf = DecisionTreeClassifier(criterion='entropy', max_features=11, random_state=i)
        clf.fit(X, y)
        trees.append(clf)
    return trees

def predict_ensemble(trees, df):
    """
    对输入的 DataFrame 使用 ensemble 模型预测标签。
    每棵树返回类别预测概率，最后取平均概率，并选择概率最大的类别作为预测结果。
    返回一个数组，表示每个样本预测的 label。
    """
    X = get_features(df)
    # 获取所有树的预测概率，列表中每个元素形状为 (n_samples, n_classes)
    prob_list = [clf.predict_proba(X) for clf in trees]
    
    # 求各棵树预测概率的平均值
    avg_prob = np.mean(prob_list, axis=0)
    
    # 获取类别标签（假设所有树的 classes_ 均一致）
    classes = trees[0].classes_
    
    # 对每个样本，选择预测概率最大的类别
    predicted_labels = classes[np.argmax(avg_prob, axis=1)]
    return predicted_labels

def get_predicted_label_sequence(trees, df):
    """
    生成预测 label 序列，直接调用 predict_ensemble，并将结果转换为 Pandas Series，
    返回与输入 DataFrame 索引对齐的预测标签序列。
    """
    pred_labels = predict_ensemble(trees, df)
    return pd.Series(pred_labels, index=df.index)

def evaluate_ensemble(trees, df):
    """
    在输入的 DataFrame 上评估 ensemble 模型的分类准确率。
    """
    y_true = df['label'].astype(int)
    y_pred = predict_ensemble(trees, df)
    acc = accuracy_score(y_true, y_pred)
    return acc
