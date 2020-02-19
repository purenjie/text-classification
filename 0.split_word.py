# coding=utf-8
import os
import jieba
import jieba.posseg  # 需要另外加载一个词性标注模块
import pickle

"""
存储所有文本文件路径
"""
99538 + 101839 + 104827 + 107592 + 92240 + 102082 + 107287 + 123096 + 108207 + 95613

culture = "/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/culture_f"  # 99999 个  有效 99538 个
medical = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/medical_f'  # 101847 个 有效 101839 个

auto = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/auto_f'  # 106000 个 有效 104827 个
dressing = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/dressing_f'  # 107592 个 有效 107592 个
ent = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/ent_f'  # 92248 个 有效 92240 个
finance = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/finance_f'  # 102100 个 有效 102082 个

life = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/life_backup'  # 107613 个 有效 107287 个
millitary = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/mil_f'  # 123161 个 有效 123096 个
social = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/social_f'  # 108226 个 有效 108207 个
sports = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/sports_f'  # 95612 个 有效 95613 个

# tech = '/media/solejay/文档/purenjie/研究生/数据仓库和数据挖掘/data/tech_f'


file_lists = []  # 保存文本路径

folder = os.walk(finance)  # 要填
for path, dir_list, file_list in folder:
    for file_name in file_list:
        file_lists.append(os.path.join(path, file_name))

"""
分词、取名词
"""
stop_words = '../数据预处理/stop_words.txt'
userdict = '../数据预处理/userdict.txt'
category_data = []  # 类别所有数据
count = 0
for article_path in file_lists:

    with open(article_path, 'r') as f:
        article = f.read()  # str

    # 加载停用词表
    stop = [line.strip() for line in open(stop_words).readlines()]

    # 导入自定义词典
    jieba.load_userdict(userdict)

    words = jieba.posseg.cut(article)  # 生成器对象
    word_attribute = ['n', 'ng', 'nr', 'ns', 'nt', 'nt', 'nz',
                      'v', 'vg', 'vd', 'vi', 'vn', 'a', 'm']  # 词语属性

    split_words = []  # 一篇文章的分词结果
    for word in words:
        if word not in stop:  # 去停用词
            for attribute in word_attribute:
                if word.flag == attribute:
                    split_words.append(word.word)
                    break

    if len(split_words) > 0:  # 分词长度不为 0 则添加该篇文本
        category_data.append(" ".join(split_words))

    print(count)
    count = count + 1

print('类别数目：', len(category_data))

label = ['finance'] * len(category_data)  # 要填
cate_file = '../pre_processed_data/finance_plus_vam_stopwords.pkl'  # 要填
with open(cate_file, 'wb') as f:
    pickle.dump(category_data, f)
    pickle.dump(label, f)
