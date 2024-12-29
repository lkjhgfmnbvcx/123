# 時間序列預測
本教學是使用 TensorFlow 進行時間序列預測的簡介。它建構了幾種不同樣式的模型，包括卷積神經網路 (CNN) 和循環神經網路 (RNN)。

本教學包括兩個主要部分，每個部分包含若干小節：

預測單一時間步驟：
單一特徵。
所有特徵。
預測多個時間步驟：
單次：一次做出所有預測。
自迴歸：一次做出一個預測，並將輸出饋送回模型。

<h2>安裝</h2>
![image](https://github.com/lkjhgfmnbvcx/123/blob/main/1.png)
天氣資料集
本教學使用馬克斯普朗克生物地球化學研究所記錄的[天氣時間序列資料集。

此資料集包含了 14 個不同特徵，例如氣溫、氣壓和濕度。自 2003 年起，這些數據每 10 分鐘就會被收集一次。為了提高效率，您將只使用 2009 至 2016 年之間收集的資料。資料集的這一部分由 François Chollet 為他的 Deep Learning with Python 一書所準備。
