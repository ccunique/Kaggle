# part of 2nd place solution: lightgbm model with private score 0.29124 and public lb score 0.28555
# https://www.kaggle.com/xiaozhouwang/2nd-place-lightgbm-solution
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44558

import lightgbm as lgbm
from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column(from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

cv_only = True
save_cv = True
full_train = False

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

path = os.path.join(os.getcwd(),'data/')

# 直接读数据,没有处理缺失值，原数据中所有缺失值都是用'-1'表示的
train = pd.read_csv(path+'train.csv')
train_label = train['target']
train_id = train['id']
test = pd.read_csv(path+'test.csv')
test_id = test['id']

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

y = train['target'].values
drop_feature = ['id','target']

X = train.drop(drop_feature,axis=1)
feature_names = X.columns.tolist()
cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
num_features = [c for c in feature_names if ('cat' not in c and 'calc' not in c)]

train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)
num_features.append('missing')

for c in cat_features:
    le = LabelEncoder()
    le.fit(train[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])


# 注意：labelencode/onehot应该针对所有数据（训练+测试数据）fit更合适，这里的处理不是很合适
# sklearn的OneHotEncoder默认是返回稀疏矩阵,不能直接查看和操作，需要利用scipy库中的sparse来操作
enc = OneHotEncoder()
enc.fit(train[cat_features])
# 把onehot后稀疏矩阵单独拿出来，最后再和其他特征合并
X_cat = enc.transform(train[cat_features])
X_t_cat = enc.transform(test[cat_features])

ind_features = [c for c in feature_names if 'ind' in c]
count=0
for c in ind_features:
    if count==0:
        train['new_ind'] = train[c].astype(str)+'_'
        test['new_ind'] = test[c].astype(str)+'_'
        count+=1
    else:
        train['new_ind'] += train[c].astype(str)+'_'
        test['new_ind'] += test[c].astype(str)+'_'


# 为每个种类特征添加一个计数特征
cat_count_features = []
for c in cat_features+['new_ind']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)

# 注意，onehot了所有原始的种类特征，但是每个种类特征伴随的计算特征没有onehot
train_list = [train[num_features+cat_count_features].values,X_cat,]
test_list = [test[num_features+cat_count_features].values,X_t_cat,]


# 包含了稀疏矩阵，需要通过ssp(sparse)来操作
# csr为稀疏矩阵存储的一种格式，CSR格式常用于读入数据后进行稀疏矩阵计算。更多格式介绍见https://www.cnblogs.com/xbinworld/p/4273506.html
X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()

learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
# 其中一些参数是做实验的时候留下的，选择"boosting_type": "gbdt"时这些参数是不起作用的，可以删除这些参数
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
           "max_bin": 256,
          "feature_fraction": feature_fraction,
          "verbosity": 0, #verbose
          "drop_rate": 0.1,  # "boosting_type": "dart"时的参数，不起作用，可删除
          "is_unbalance": False,
          "max_drop": 50,  # "boosting_type": "dart"时的参数，不起作用，可删除
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
          }

x_score = []
final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))

# 固定训练集+验证集划分方式，选了16个不同种子的lgb平均
for s in range(16):
    cv_train = np.zeros(len(train_label))
    cv_pred = np.zeros(len(test_id))

    params['seed'] = s

    if cv_only:
        kf = kfold.split(X, train_label)

        best_trees = []
        fold_scores = []

        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
            dtrain = lgbm.Dataset(X_train, label_train)
            dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
            bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
            best_trees.append(bst.best_iteration)
            cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
            cv_train[validate] += bst.predict(X_validate)

            score = Gini(label_validate, cv_train[validate])
            print(score)
            fold_scores.append(score)

        cv_pred /= NFOLDS
        final_cv_train += cv_train
        final_cv_pred += cv_pred

        print("cv score:")
        print(Gini(train_label, cv_train))
        print("current score:", Gini(train_label, final_cv_train / (s + 1.)), s+1)
        print(fold_scores) #可以顺便输出一下每个模型CV的平均得分及方差
        print(best_trees, np.mean(best_trees))
        print()

        x_score.append(Gini(train_label, cv_train))

print(x_score)
# pd.DataFrame({'id': test_id, 'target': final_cv_pred / 16.}).to_csv('./solutions_2nd/lgbm3_pred_avg.csv', index=False)
# pd.DataFrame({'id': train_id, 'target': final_cv_train / 16.}).to_csv('./solutions_2nd/lgbm3_cv_avg.csv', index=False)
