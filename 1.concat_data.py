import pickle
import csv
import pandas as pd

category = ['culture', 'medical_plus_v', 'automobile', 'dressing', 'entertainment', 'finance_plus_vam', 'life', 'military',
            'social_plus_v', 'sports']


def concat_data(data, label, cate):
    cate_path = '../pre_processed_data/%s.pkl' % cate
    with open(cate_path, 'rb') as f:
        cate_data = pickle.load(f)
        cate_label = pickle.load(f)
    print('%s读取完毕' % cate)
    data.extend(cate_data)
    label.extend(cate_label)
    print('%s添加完毕' % cate)

data = []
label = []

for cate in category:
    concat_data(data, label, cate)

print(len(data))  # 1042321 1042323 1042326 1042326
print(len(label))  # 1042321 1042323 1042326 1042326

with open('../pre_processed_data/concat_10_data_plus_vam.pkl', 'wb') as f:
    pickle.dump(data, f)
    pickle.dump(label, f)

# """
# 拼接词典
# """
# def concat_set(data_dict, cate):
#     cate_path = '../dict/%s_dict.pkl' % cate
#     with open(cate_path, 'rb') as f:
#         cate_dict = pickle.load(f)
#     print('%s读取完毕' % cate)
#     # print(type(cate_dict))
#     print(len(cate_dict))
#     data_dict = data_dict | cate_dict
#     print('%s添加完毕' % cate)
#     print(len(data_dict))
#     return data_dict
#
#
# data_dict = set()
# for cate in category:
#     data_dict = concat_set(data_dict, cate)
#
# dict_path = '../dict/data_dict.pkl'
# with open(dict_path, 'wb') as f:
#     pickle.dump(data_dict, f)










