# 거래금액
# 거래금액 데이터 컬럼명 변경
train_data.rename(columns = {'거래금액(만원)':'거래금액'}, inplace=True)
# 거래금액 , 제거 후 숫자형 변환
train_data['거래금액'] = train_data['거래금액'].str.replace(',','').astype(int)


## 계약년월
# 계약년도와 계약월 컬럼 생성(EDA)
train_data['계약년도'] = train_data['계약년월'].map(lambda x : int(str(x)[:4]))
train_data['계약월'] = train_data['계약년월'].map(lambda x : int(str(x)[4:]))
train_data
