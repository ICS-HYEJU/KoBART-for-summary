# KoBART-for-summary
## DATA
* [AIHub 문서요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=97)
* 데이터 구성 : 신문기사, 사설, 법률 (각 train/valid josn파일 존재)
* 데이터 추출에 용이하게 tsv로형태로 데이터를 변환함(./data/split/py)  
* Data 구조
   | text | summary |
   |------|----------|
   | 원문 | 생성요약 label |
* 데이터 정제 시, 다음과 같은 유의사항이 존재
     *  생성요약 label인 abstractive가 None인 데이터 존재

        (
