import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

sent = ("휴일 인 오늘 도 서쪽 을 중심 으로 폭염 이 이어졌는데요, 내일 은 반가운 비 소식 이 있습니다.",
        "폭염 을 피해서 휴일 에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니다.")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sent)  # 문장 벡터화 진행

idf = tfidf_vectorizer.idf_
print(dict(zip(tfidf_vectorizer.get_feature_names(), idf)))  # 각 수치에 대한 값 시각화

# 자카드 유사도
from sklearn.metrics import jaccard_score

# jaccard_score(tfidf_matrix[0:1], tfidf_matrix[1:2])
jaccard_score(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 2]), average=None)

# 코사인 유사도
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# 유클리디안 유사도
from sklearn.metrics.pairwise import euclidean_distances

euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])


# 유클리디안 유사도는 1보다 큰 값이 나옴 --> 0, 1 정규화를 해야 다른 유사도와 비교 가능

# 정규화
def l1_normalize(v):
    norm = np.sum(v)
    return v / norm


tfidf_norm_l1 = l1_normalize(tfidf_matrix)
euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])

# 맨해튼 유사도
# 동일하게 정규화해서 사용해야 함
from sklearn.metrics.pairwise import manhattan_distances

manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
