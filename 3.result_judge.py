from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pickle
import numpy as np
from sklearn.metrics import classification_report


# 绘制混淆矩阵

def plot_confusion_matrix(cm, classes, savename=None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../result/%s.png' % savename)
    plt.show()


result_file = '0.9352_NB_bow_finance_plus_vam_gridsearchCV_37'  # 要改

with open('../result/%s.pkl' % result_file, 'rb') as f:
    y_true = pickle.load(f)
    y_pred = pickle.load(f)
    train_time = pickle.load(f)
    test_time = pickle.load(f)
    x_train_num, x_train_dimension = pickle.load(f)
    x_test_num, x_test_dimension = pickle.load(f)

# x_train_num, x_train_dimension = x_train_shape
# x_test_num, x_test_dimension = x_test_shape


labels = ['culture', 'medical', 'automobile', 'dressing', 'entertainment', 'finance', 'life', 'military', 'social',
          'sports']

accuracy = accuracy_score(y_true, y_pred)  # 准确率：分类正确的样本占总样本个数的比例
precision = precision_score(y_true, y_pred, average='weighted')  # 精确率：模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
recall = recall_score(y_true, y_pred, average='weighted')  # 召回率：实际为正的样本中被预测为正的样本所占实际为正的样本的比例
f1 = f1_score(y_true, y_pred, average='weighted')  # f1值：精确率和召回率的调和平均值

# 保存模型评价结果
with open('../result/%s.txt' % result_file, 'w') as f:
    s1 = 'accuracy：%.4f' % accuracy + '\n'
    s2 = 'precision：%.4f' % precision + '\n'
    s3 = 'recall：%.4f' % recall + '\n'
    s4 = 'f1：%.4f' % f1 + '\n'
    s5 = 'train_time：%.4f s' % train_time.total_seconds() + '\n'
    s6 = 'test_time：%.4f s' % test_time.total_seconds() + '\n'
    s7 = '训练集样本数：%d' % x_train_num + '\n'
    s8 = '测试集样本数：%d' % x_test_num + '\n'
    s9 = '训练集维度：%d' % x_train_dimension + '\n'
    s10 = '测试集维度：%d' % x_test_dimension + '\n'

    s = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
    f.write(s)


# 打印混淆矩阵
matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(matrix, classes=labels, savename=result_file)




