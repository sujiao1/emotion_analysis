import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba

data = pd.read_csv("data.csv")
data.head()

stopword_path = r"stop_words.txt"
stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]


def segment(data):
    result_data = [i for i in jieba.cut(data.strip()) if i not in stopwords and i != " "]
    result_data = " ".join(result_data)
    return result_data


data["segment"] = data["text"].apply(segment)
data.head()


# data["segment"].tolist() 可以产生segment的列表，["手机 左下角...","照相 功能 手机...","给 老人 买 字体 清晰 声音..."]
# 制作词频字典{word1:feq1, word2:feq2, word3:feq3.....},筛选出出现次数大于3的字典
# TODO

word_dict = {}

for sent in data["segment"].tolist():
    for i in sent.split(" "):
        if i in word_dict.keys():
            word_dict[i]+=1
        else:
            word_dict[i]=1
word_dict = {key:value for key,value in word_dict.items() if value>3}
print("词典长度：")
print(len(word_dict.keys()))
print(len(word_dict.values()))

def screen_word(data):
    data = [i for i in data.split(" ") if i in word_dict.keys()]
    data = " ".join(data)
    return data

data["screen_segment"] = data["segment"].apply(screen_word)
data.head()

with open("dict/neg.txt","r",encoding='gb18030', errors='ignore') as f:
    negword = f.readlines()
negdict = [i.strip() for i in negword if i !='']
with open("dict/pos.txt","r",encoding='gb18030', errors='ignore') as f:
    posword = f.readlines()
posdict = [i.strip() for i in posword if i !='']
with open("dict/no.txt","r",encoding='gb18030', errors='ignore') as f:
    noword = f.readlines()
nodict = [i.strip() for i in noword if i !='']
with open("dict/plus.txt","r",encoding='gb18030', errors='ignore') as f:
    plusword = f.readlines()
plusdict = [i.strip() for i in plusword if i !='']


def check_emotion(data):
    """
        Args:
            data: str 一条分好词的评论，词之间用空格分开，例如：“手机 左下角 有 有 瑕 丝 质量 问题 手机”
        Return:
            score: int 当前这句话的得分

    """
    score = 0
    data = data.split(" ")
    for i in range(len(data)):
        if data[i] in negdict:
            if i > 0 and data[i - 1] in nodict:
                score = score + 1
            elif i > 0 and data[i - 1] in plusdict:
                score = score - 2
            else:
                score = score - 1
        elif data[i] in posdict:
            if i > 0 and data[i - 1] in nodict:
                score = score - 1
            elif i > 0 and data[i - 1] in plusdict:
                score = score + 2
            elif i > 0 and data[i - 1] in negdict:
                score = score - 1
            elif i < len(data) - 1 and data[i + 1] in negdict:
                score = score - 1
            else:
                score = score + 1
        elif data[i] in nodict:
            score = score - 0.5
    return score

data["emotion"] = data["screen_segment"].apply(check_emotion)
data.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
word_list = data["screen_segment"].tolist()
data_tfidf = tfidf_vec.fit_transform(word_list)

print("tfidf维度：")
print(np.shape(data_tfidf.todense()))

emotion_feature = np.expand_dims(np.array(data["emotion"].tolist()),axis=1)
print(np.shape(emotion_feature))

transform_feature = np.concatenate([data_tfidf.todense(), emotion_feature],axis=1)
print("数据维度：")
print(np.shape(transform_feature))

print("标签维度：")
label = data["label"].tolist()
label = np.array(label)
print(np.shape(label))

from sklearn.model_selection import train_test_split

train_data,test_data,train_label,test_label = train_test_split(transform_feature,label,train_size=0.8,test_size=0.2,
                                                               random_state=0,shuffle=True)
print("训练集数据维度：")
print(np.shape(train_data))

print("测试集数据维度：")
print(np.shape(test_data))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

model = LogisticRegression()
model.fit(train_data,train_label)

print("---------------------训练集-----------------------")
train_pred_result = model.predict(train_data)
accuracy = accuracy_score(y_pred=train_pred_result,y_true=train_label)
precision = precision_score(y_pred=train_pred_result,y_true=train_label,average=None)
recall = recall_score(y_pred=train_pred_result,y_true=train_label,average=None)
cm = confusion_matrix(y_pred=train_pred_result,y_true=train_label)
print("正确率：",accuracy)
print("精确率：",precision)
print("召回率：",recall)
print(cm)

print("---------------------测试集-----------------------")
pred_result = model.predict(test_data)
accuracy = accuracy_score(y_pred=pred_result,y_true=test_label)
precision = precision_score(y_pred=pred_result,y_true=test_label,average=None)
recall = recall_score(y_pred=pred_result,y_true=test_label,average=None)
cm = confusion_matrix(y_pred=pred_result,y_true=test_label)
print("正确率：",accuracy)
print("精确率：",precision)
print("召回率：",recall)
print(cm)


def check(origin_data):
    data = " ".join([i for i in list(jieba.cut(str(origin_data))) if i not in stopwords])
    data_tfidf = [data]
    result_vecter = tfidf_vec.transform(data_tfidf).todense()

    num = check_emotion(data)
    num = np.expand_dims([num], axis=1)

    result_vecter = np.concatenate([result_vecter, num], axis=1)
    pred_result = model.predict(result_vecter)
    dict_map = {0: "这是一条差评", 1: "这是一条中评", 2: "这是一条好评"}
    return dict_map[pred_result[0]]


in_sent = input()
judge_result = check(in_sent)
print(judge_result)