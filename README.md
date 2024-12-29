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
<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/1.png">

天氣資料集
本教學使用馬克斯普朗克生物地球化學研究所記錄的[天氣時間序列資料集。

此資料集包含了 14 個不同特徵，例如氣溫、氣壓和濕度。自 2003 年起，這些數據每 10 分鐘就會被收集一次。為了提高效率，您將只使用 2009 至 2016 年之間收集的資料。資料集的這一部分由 François Chollet 為他的 Deep Learning with Python 一書所準備。

<h2>天氣數據集</h2>
本教學使用馬克斯普朗克生物地球化學研究所記錄的[天氣時間序列資料集。

此資料集包含了 14 個不同特徵，例如氣溫、氣壓和濕度。自 2003 年起，這些數據每 10 分鐘就會被收集一次。為了提高效率，您將只使用 2009 至 2016 年之間收集的資料。資料集的這一部分由 François Chollet 為他的 Deep Learning with Python 一書所準備。
<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/2.png">

本教學僅處理每小時預測，因此先從 10 分鐘間隔到 1 小時對資料進行下採樣：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/3.png">

讓我們看一下數據。下面是前幾行：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/4.png">

以下是一些特徵隨時間的演變：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/5.png">

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/6.png">

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/7.png">

<h2>檢查和清理</h2>
接下來，來看看資料集的統計數據：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/8.png">

風速
值得注意的一件事是風速 (wv (m/s)) 的 min 值和最大值 (max. wv (m/s)) 列。這個 -9999 可能是錯的。

有一個單獨的風向列，因此速度應大於零 (>=0)。將其替換為零：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/9.png">

<h2>特徵工程</h2>
在潛心建立模型之前，請務必了解資料並確保傳遞格式正確的資料。

風
資料的最後一列 wd (deg) 以度為單位給出了風向。角度不是很好的模型輸入：360° 和 0° 應該會彼此接近，並且平滑換行。如果不吹風，方向則無關緊要。

現在，風資料的分佈狀況如下：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/10.png">

但是，如果將風向和風速列轉換為風向量，模型將更容易解釋：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/11.png">

模型正確解釋風向量的分佈要簡單得多：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/12.png">

時間
同樣，Date Time 列非常有用，但不是以這種字串形式。首先將其轉換為秒：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/13.png">

與風向類似，以秒為單位的時間不是有用的模型輸入。作為天氣數據，它​​有清晰的每日和每年週期性。可以透過多種方式處理週期性。

您可以透過使用正弦和餘弦變換為清晰的“一天中的時間”和“一年中的時間”信號來獲得可用的信號：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/14.png">

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/15.png">

這使模型能夠存取最重要的頻率特徵。在這種情況下，您提前知道了哪些頻率很重要。

如果您沒有該資訊，則可以透過使用快速傅立葉變換提取特徵來確定哪些頻率重要。要檢驗假設，以下是溫度隨時間變化的 tf.signal.rfft。請注意 1/year 和 1/day 附近頻率的明顯峰值：

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/16.png">

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/17.png">

<h2>拆分數據</h2>
您將使用 (70%, 20%, 10%) 拆分出訓練集、驗證集和測試集。請注意，在拆分前資料沒有隨機打亂順序。這有兩個原因：

確保仍然可以將資料切入連續樣本的視窗。
確保訓練後在收集的數據上對模型進行評估，驗證/測試結果更加真實。

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/18.png">

<h2>歸一化數據</h2>
在訓練神經網路之前縮放特徵很重要。歸一化是進行此類縮放的常見方式：減去平均值，然後除以每個特徵的標準差。

平均值和標準偏差應僅使用訓練資料進行計算，從而使模型無法存取驗證集和測試集中的值。

有待商榷的是：模型在訓練時不應存取訓練集中的未來值，以及應該使用移動平均數來進行此類規範化。這不是本教學的重點，驗證集和測試集會確保我們獲得（某種程度上）可靠的指標。因此，為了簡單起見，本教學使用的是簡單平均數。

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/19.png">

在看這些特徵的分佈。部分特徵的尾部確實很長，但沒有類似 -9999 風速值的明顯誤差。

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/20.png">

<img src="https://github.com/lkjhgfmnbvcx/123/blob/main/21.png">

<h2>資料視窗化</h2>
本教程中的模型將基於來自數據連續樣本的窗口進行一組預測。
輸入窗口的主要特徵包括：

<li>輸入和標籤窗口的寬度（時間步驟數量）。</li>
<li>它們之間的時間偏移量。</li>
<li>用作輸入、標籤或兩者的特徵。</li>

本教程構建了各種模型（包括線性、DNN、CNN 和 RNN 模型），並將它們用於以下兩種情況：

<li>單輸出和多輸出預測。</li>
<li>單時間步驟和多時間步驟預測。</li>

本部分重點介紹實現數據窗口化，以便將其重用到上述所有模型。
根據任務和模型類型，您可能需要生成各種數據窗口。下面是一些示例：
<ol>
<li>例如，要在給定 24 小時歷史記錄的情況下對未來 24 小時作出一次預測，可以定義如下窗口：</li>
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
<img src="">
