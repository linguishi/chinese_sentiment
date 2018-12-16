### LSTM 中文情感分析

#### 语料
语料的选择为 *谭松波老师的评论语料*，正负例各2000。

解压 `corpus.zip` 后运行 
```sh
python FixCorpus.py
```
将原本`gb2312`编码文件转换成`utf-8`编码的文件。

语料的具体测试参考 [`CorpusTests.py`](https://github.com/linguishi/lstm-sentiment/blob/master/CorpusTests.ipynb)

#### 词向量
本实验使用开源词向量[*chinese-word-vectors*](https://github.com/Embedding/Chinese-Word-Vectors)

选择知乎语料训练而成的Word Vector，[点击下载](https://pan.baidu.com/s/1OQ6fQLCgqT43WTwh5fh_lg)

词向量的相关测试，可参考 [`CNWordVecTest.ipynb`](https://github.com/linguishi/lstm-sentiment/blob/master/CNWordVecTest.ipynb)

#### 建模、训练、测试
`main.py` 或者 `Main.ipynb`

#### 一些测试结果
```
Accuracy:0.857500

酒店设施不是新的，服务态度很不好
是一例负面评价 output=0.44

酒店卫生条件非常不好
是一例负面评价 output=0.26

床铺非常舒适
是一例正面评价 output=0.65

房间很凉，不给开暖气
是一例负面评价 output=0.23

房间很凉爽，空调冷气很足
是一例正面评价 output=0.50

酒店环境不好，住宿体验很不好
是一例负面评价 output=0.26

房间隔音不到位
是一例负面评价 output=0.29

晚上回来发现没有打扫卫生
是一例负面评价 output=0.33

因为过节所以要我临时加钱，比团购的价格贵
是一例负面评价 output=0.17
```

