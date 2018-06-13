# DGADETEC 文档

**dgadetec** 是用于DGA域名检测的模块，支持高并发，初次加载模型，以及其他配置文件时速度很慢，但只需加载一次到内存之后就很快。
什么是DGA请参考[维基百科](http://www.baidu.com). 

**主体思路**： 以机器学习为依托，采用特征提取，辅以模型融合为主要方法，训练模型。提供接口，识别域名是否为DGA域名，正常域名以 1 为标记，dga域名以 0 为标记

-------------------

[TOC]

##编译安装

模块所需依赖如下：
- numpy
-  pandas
- scikit-learn
- python2.7

正常安装所需依赖后，公司内部下载获取dgadetec源码，解压后进入文件目录 
安装过程因为涉及部署模型，所以在模型训练部署阶段速度会非常的慢
>cd dgadetec
> python setup.py install

## 使用接口预测DGA
接口只有一个 [predict](#组织结构), 参数也只有一个，来自[detector](#组织结构)模块：
> *predict(domain_list)*
> - **domain_list**:
>  需要是一个 1D 的列表或是numpy格式的向量, 里面的元素为字符串格式的域名，一个或者多个域名
>类似于 ['www.baudu.com']或['www.'google.com', www.dkasjdkasd.com]
>- **return**
>返回一个只包含0和1的列表，1代表识别为正常域名，0代表识别为dga域名，其位置下标对应输入域名的下标 
```python
>>from dgadetec import detector
>>domain = ['www.baidu.com','www.dmsak2jddd.com']
>>pred_y = detector.predict(domain)
....
[1, 0]
``` 
## 组织结构
> \dgadetec
>  ..  \resource
>  ....    *.npy
>  ..  \models
>  .... \*.pkl
>  .... \*.npy
>  *.py


整个模块包含以下
**代码**：位于[dgadetec](#)下的Python脚本
- *settings.py*:  整个库所需的配置文件，主要为所需数据文件的绝对路径
- *prepare_model.py*: 
 1. 训练生成整个库所需提前准备的，用于提取dga特征的模型、数据文件，预测模型。
 2. 后期需要更新这些模型的函数也包含在其中
- *feature_extractor.py* : 特征提取模块，所有的特征将用这个模块提取
- *detector.py*：预测模块，上面提到的接口在这里，出预测接口外，还含有一些额外的输出输出错误检查处理机制
- *cluster.py*: 聚类模块，用于获取正常域名的聚类中心，所获得的正常中心域名（10个）将   用于加速计算jarcarrd系数。此模块在提取特征时将不被使用，只是在更换了正常域名集后需要更新已有的中心域名

**数据文件** 包含两部分 [models](#) 和 [resource](#)
除了一部分为外部数据外，所有这些数据将由 *prepare_model* 提前生成，在使用这个模块的时候，你需要先生成所需文件
- 外部数据：所有外部数据包含于resource：
  1. train.npy 训练预测模型的数据，更新模型只需覆盖这个文件  
  2. aleax100k.npy 正常域名集，用于cluster中，提取正常中心域名
  3. words.csv n_grame特征 所需数据 
 

- 内部数据：除了外部数据意外，其他都是内部生成数据，包括models/ 里的数据，很多大型数据是不需要被绑定安装的（安装了那些文件请看setup.py），它们只是在训练模型的时候会被使用到。
-

## 提取特征
#### 1. 元音字母 located in [count_aeiou](#)
  >提取域名中的元音字母数量、元音字母数所占域名总字符数的比例、域名长度
  
#### 2. 域名字符去重后的长度，及此长度所占域名长度的比例 located in [unique_char_rate](#)
> 域名去重后，计算其去重后的长度，以及这个长度所占域名原长度的比例
  
#### 3. 域名熵
  >  $$entropy= \sum_{i=1}^{n}{P_i}*log_2P_i$$
  >  其中$P_i$为域名中某个字母与域名长度的比值，$n$为不重复字母的集合，$n\in(a,b,c....l,m,n)$
  >  e.g. www.baidu.com 中字母w的概率为$P_w=\frac{3}{13}$

#### 4. n_grame 
此处所采用的grame范围为(3,5)

- 通过正常域名训练出一个string转词向量模型(CountVector):

**Positive Domains Vector **(Matrix PDC)
 Domains| grame1 | grame2 | grame 3|.... 
- | :-: | :-:|  -: 
domainA| 1| 0|1| ...
domainB | 1 | 1|0| ...
domainC| 0 | 0 |0|...

并计算其矩阵对数和（纵向）

- 利用PDC获取输入域名的转换矩阵：

**Tartget Domain  **(Matrix TD)
 Domains| grame1 | grame2 | grame 3|.... 
- | :-: | :-:|  -: 
domainX| 1| 0|1| ...
domainY | 1 | 1|0| ...
domainZ| 0 | 0 |0|...
domainM| 0 | 1 |0|...

- 计算匹配程度
$MatchPositiveDomain = PDC*TD^T$

与上述类似的，还有词典匹配，将PositiveDomain换成字典就可以了，字典文件为[words.csv](#数据文件) 








#### 5. jarccard系数
> 两个域名A,B 之间的jarccard 系数为$$jarccard(A, B)=\frac{{A}\bigcap{B}}{{A}\bigcup{B}}$$
>这里假设我们所要提取的特征的域名为A，正常域名为B，整个过程中，A需要和大量的正常域名B做系数计算， 再将这些系数相加取均值作为jarccard系数，公式如下
>$$AvgJarccarc_a =\frac{\sum_{i=1}^{N}{jarccard(a,B_i)}}{M}, b\in B, a\in A$$
>$N$ 等于域名集合$A$的长度，$M$等于域名集合$B$的长度


6. **HMM** 隐性马尔科夫系数
> 这部分的特征需要预先训练一个联合高斯分HMM（隐性马尔可夫模型），并预测域名的系数。训练过程中，数据使用正常域名而非DGA域名，此处使用的是Alexa100k的域名，因为机器资源以及训练时间问题，只选择了其中的500个域名<shuffer后>
> 其中，HMM系数越接近0则越贴合模型，表示域名更接近正常人所取的名称，而非机器。也就是读起来很顺口

#### 7. KL离散系数(K-L divergence)
两个域名$P,Q$之间的K-L系数表示为：
$$D_{KL}(P, Q)=\sum_{i=1}^{n}P_i*log\frac{P_i}{Q_i}$$
此处$n\in(a, b, c, ..., z...1,2,3..9)\in P$
修改后的K-L离散系数特征公式为：
$$D_{sym}(P, Q)=\frac{1}{2}\big(D_{KL}(P, Q) + D_{KL}(Q,P)\big)$$
 具体理论请参考这篇[paper](http://eprints.networks.imdea.org/67/1/Detecting_Algorithmically_Generated_Malicious_Domain_Names_-_2010_EN.pdf)

## 更新数据，更新模型
##### 对于更新模型数据：替换在[resource](#)录下你想要更新的文件，其中
- **train.npy**: 为模型所依赖的训练数据，格式为：numpy存储格式npy，2D矩阵（域名,标签），域名需要包括正常域名和DGA域名，标签为 0或1。
- **aleax100k.npy**：正常域名集合，无标签只有域名，1D向量
- **words.csv**:单词文件，不需要替换

## 参考论文
papers目录下的论文，大部分参考
- *“dga_relarion_paper_implementation“*
