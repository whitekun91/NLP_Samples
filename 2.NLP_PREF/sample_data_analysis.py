import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# wordcloud
from wordcloud import WordCloud, STOPWORDS


def directory_data(directory):
    data = {}
    data["review"] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), 'r', encoding='utf-8') as file:
            data['review'].append(file.read())

    return pd.DataFrame.from_dict(data)


def data(directory):
    pos_df = directory_data(os.path.join(directory, "pos"))
    neg_df = directory_data(os.path.join(directory, "neg"))

    pos_df['sentiment'] = 1
    neg_df['sentiment'] = 0

    return pd.concat([pos_df, neg_df])


if __name__ == '__main__':
    # Data load - imdb
    data_set = tf.keras.utils.get_file(
        fname="imdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = data(os.path.join(os.path.dirname(data_set), 'aclImdb', 'train'))
    test_df = data(os.path.join(os.path.dirname(data_set), 'aclImdb', 'test'))

    print(train_df.head())

    reviews = list(train_df['review'])

    print(reviews)

    # data analysis
    # 문자열 문장 리스트를 토크나이징
    tokenized_reviews = [r.split() for r in reviews]

    # 토크나이징된 리스트에 대한 각 길이를 저장
    review_len_by_token = [len(t) for t in tokenized_reviews]

    # 토크나이징된 것을 붙여서 음절의 길이를 저장
    review_len_by_eumjeol = [len(s.replace(' ', '')) for s in reviews]

    plt.figure(figsize=(12, 5))
    plt.hist(review_len_by_token, bins=50, alpha=0.5, color='r', label='word')
    plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color='b', label='alphabet')
    plt.yscale('log', nonposy='clip')

    plt.title('Review Length Histogram')
    plt.xlabel('Review Length')
    plt.ylabel('Number of Reviews')

    print('문장 최대길이: {}'.format(np.max(review_len_by_token)))
    print('문장 최소길이: {}'.format(np.min(review_len_by_token)))
    print('문장 평균길이: {:.2f}'.format(np.mean(review_len_by_token)))
    print('문장 길이 표준편차: {:.2f}'.format(np.std(review_len_by_token)))
    print('문장 중간길이: {}'.format(np.median(review_len_by_token)))
    # 사분위의 대한 경우는 0~100 스케일로 되어있음
    print('제 1 사분위 길이: {}'.format(np.percentile(review_len_by_token, 25)))
    print('제 3 사분위 길이: {}'.format(np.percentile(review_len_by_token, 75)))


    plt.figure(figsize=(12, 5))
    # 박스플롯 생성
    # 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
    # labels: 입력한 데이터에 대한 라벨
    # showmeans: 평균값을 마크함

    plt.boxplot([review_len_by_token],
                labels=['token'],
                showmeans=True)

    plt.figure(figsize=(12, 5))
    plt.boxplot([review_len_by_eumjeol],
                labels=['Eumjeol'],
                showmeans=True)

    # wordcloud
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=800, height=600).generate(' '.join(train_df['review']))

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


    sentiment = train_df['sentiment'].value_counts()
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_df['sentiment'])
