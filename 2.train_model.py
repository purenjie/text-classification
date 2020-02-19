import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
import datetime


with open('../pre_processed_data/concat_10_data_plus_vam.pkl', 'rb') as f:
    data = pickle.load(f)  # 1042326
    label = pickle.load(f)  # 1042326

x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=42, test_size=0.7)


bow = CountVectorizer()

print(len(x_train))  # 312697
print(len(x_test))  # 729629

x_train = bow.fit_transform(x_train)  # csr_matrix (312697, 616910)
x_test = bow.transform(x_test)  # csr_matrix (729629, 616910)


vec_name = 'bow_finance_plus_vam_37'

# with open('../vector/%s.pkl' % vec_name, 'wb') as f:
#     pickle.dump(x_train, f)
#     pickle.dump(y_train, f)
#     pickle.dump(x_test, f)
#     pickle.dump(y_test, f)

with open('../vector/%s.pkl' % vec_name, 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    x_test = pickle.load(f)
    y_test = pickle.load(f)

x_train_shape = x_train.shape
x_test_shape = x_test.shape


print('读取完毕')

"""
朴素贝叶斯
"""
# classifier = MultinomialNB()
# classifier.fit(x_train, y_train)  # 模型训练
# y_pred = classifier.predict(x_test)  # 使用训练好的模型进行预测
# score = classifier.score(x_test, y_test)

"""
GridserachCV 调参
"""
vec_name = 'bow_finance_plus_vam_gridsearchCV_37'

train_starttime = datetime.datetime.now()
parameters = {'fit_prior': [True, False], 'alpha': np.logspace(-5, 1, num=10, base=2)}
grid_search = GridSearchCV(MultinomialNB(), parameters)
grid_search.fit(x_train, y_train)
train_endtime = datetime.datetime.now()

test_starttime = datetime.datetime.now()
y_pred = grid_search.predict(x_test)
test_endtime = datetime.datetime.now()

score = grid_search.score(x_test, y_test)
train_time = train_endtime - train_starttime
test_time = test_endtime - test_starttime
print(str(grid_search))
print(train_time)
print(test_time)


with open('../result/%.4f_NB_%s.pkl' % (score, vec_name), 'wb') as f:
    pickle.dump(y_test, f)
    pickle.dump(y_pred, f)
    pickle.dump(train_time, f)
    pickle.dump(test_time, f)
    pickle.dump(x_train_shape, f)
    pickle.dump(x_test_shape, f)

print(score)


# """
# xgboost
# """
# from sklearn import preprocessing
# import xgboost as xgb
# le = preprocessing.LabelEncoder()
# y_train = le.fit_transform(y_train)
#
# param = {'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax',
#          'num_class': 10, 'subsample': 1,
#          'colsample_bytree': 0.85}  # 参数
#
# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)
# num_round = 200
# bst = xgb.train(param, dtrain, num_round)
# y_pred = le.inverse_transform(bst.predict(dtest).astype('int64'))
#
# with open('../result/xgboost.pkl', 'wb') as f:
#     pickle.dump(y_test, f)
#     pickle.dump(y_pred, f)

