# 2021-DSAI-HW1

## data analysys
透過畫圖看出 備轉容量(MW)=淨尖峰供電能力(MW)-尖峰負載(MW)
![](https://i.imgur.com/PLgcdWB.png)
藍橘分別是淨尖峰供電能力(MW),尖峰負載(MW)
綠色的是我們要預測的項目備轉容量(MW)
可以看出淨尖峰供電能力(MW),尖峰負載(MW)是具有季節型變化的趨勢
於是我選擇SARIMA作為預測的模型，並預測淨尖峰供電能力(MW),尖峰負載(MW)，然後相減得到未來的備轉容量(MW)

## data preprocessing

對訓練資料做MinMax Normalized

## model training

SARIMA總共有七個參數，我將各種參數做組合，然後進行訓練，將預測的值與實際的值計算mean absolute percentage error，紀錄下來，然後挑出表現最好的參數，作為最終用來預測的模型的參數

## result

此圖為淨尖峰供電能力(MW)
![](https://i.imgur.com/TJTrXHp.png)
此圖為尖峰負載(MW)
![](https://i.imgur.com/FaQL4qE.png)



