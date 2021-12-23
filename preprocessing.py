# 거래금액 데이터 컬럼명 변경
train_data.rename(columns = {'거래금액(만원)':'거래금액'}, inplace=True)
# 거래금액 , 제거 후 숫자형 변환
train_data['거래금액'] = train_data['거래금액'].str.replace(',','').astype(int)
