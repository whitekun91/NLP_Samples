import konlpy
from konlpy.tag import Okt
from konlpy.corpus import kolaw
from konlpy.corpus import kobill


okt = Okt()

text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지ㅎㅎㅎ"

print(okt.morphs(text))
print(okt.morphs(text, stem=True)) # 형태소 단위로 나눈 후 어간을 추출

print(okt.nouns(text))
print(okt.phrases(text))

print(okt.pos(text))
print(okt.pos(text, join=True)) # 형태소와 품사를 붙여서 리스트화


# 한국 법률 말뭉치
print(kolaw.open('constitution.txt').read()[:20])

# 대한민국 국회 의안 말뭉치
print(kobill.open('1809890.txt').read())