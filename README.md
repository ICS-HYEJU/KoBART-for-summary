# KoBART-for-summary
## DATA
* [AIHub 문서요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=97)
* 데이터 구성 : 신문기사, 사설, 법률 ( 각 train/valid json파일 존재 )
* 데이터 추출에 용이하게 tsv로형태로 데이터를 변환함(./data/split/py)  
* Data 구조
   | text | summary |
   |------|----------|
   | 원문 | 생성요약 label |
* 데이터 정제 시, 다음과 같은 유의사항이 존재
     *  생성요약 label 'abstractive' 및 추출요약 label 'extractive'에 None인 데이터 존재
     *  데이터 내 원문인 'text'는 문단을 나타내는 list 내 sentence를 담은 list가 존재하는 2중 list 구조로,
        text list의 index 번호와 sentence의 index값이 상이함.
        
