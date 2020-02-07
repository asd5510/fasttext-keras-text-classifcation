## 关于fasttext的keras实现以及相关问题探讨

fasttext是facebook开发的一款快速文本分类的工具。工具本身有很多限制，比如只能做分类不能做回归问题，比如pooling的方式只能使用avg-pooling的方式，只能针对char级别进行embedding，无法提取训练好的embedding向量等等。
综合上述的原因，本篇探讨通过keras实现一个fasttext工具，并且探究其中涉及到的一些机器学习，文本建模相关问题的分析。

### fasttext的基本原理

fastText简而言之，就是把文档中所有词通过lookup table映射为一个embedding向量，经过avg-pooling后直接经过2层全连接层得到分类结果。本质十分类似于一个BOW模型，但由于使用了词向量因此效果更好。同时由于使用pooling的方式因此对词序不敏感，仅通过n-gram的方式捕捉一定程度的context。


 ![fasttext原理图](fasttext_struct.png"fasttext原理图")


当然，如果只用unigram的话会丢掉word order信息，所以通过加入n-gram features进行补充。fastText是对char级别生成embedding的。(这里同bert不同，bert虽然是基于字做vocab，但是每个词的向量不是简单的将字emb拼起来，而是需要完整经过bert模型前向运行得到，所以得到的词向量效果也非常好，然而fastText这种通过字向量加和成为词向量那肯定是不行的，会极大受到字面匹配程度的干扰，一个经典的例子就是fasttext得到的词向量进行kNN计算，“交易”最接近的词是“交易法”，但gensim训练的词向量则是“买卖”)。

然而Fasttext的word_ngrams参数很大影响了效率，实测当词库比较大的话如果2-gram就会非常慢了(主要是保存模型特别慢，2-gram的参数非常多了)，而实际1-gram在多迭代几次后就能达到很好的效果，多数情况下没必要上2-gram。这个同文本分类任务的特性有关：刘至远提到像文本分类这样的任务，如果是长文本，即使用BOW也能做很不错的效果。另外项亮说到文本分类大部分情况下是个简单的线性问题，因为词汇本来就是高度凝结智慧和信息量的产物了，所以多层网络没太多意义。
我的理解是文本分类任务由于很多情况下对不需要捕捉高级的语法关系，上下文语境等等，而只需要做词汇匹配就行。因此长文本下BOW不会太过稀疏，效果也会不错。

### fasttext的简单实践
facebook最早推出的是c++的版本，后续封装了python版本，实测python版本的运行效率也很高，因此本文采用的是fasttext的python版本。安装方式通过pip就能完成，这里不赘述了。我们要做一个简单的分类任务，根据用户关注的文本来预测用户的年龄，简单的demo如下：
```markdown
import fasttext as ft

input_file = 'for_tgrocery_age_preda_train'
test_file = 'for_tgrocery_age_preda_test'
output = '/tmp/classifier_comm'

# set params
dim= 30
lr=0.02
epoch= 15
min_count= 1
word_ngrams= 2
bucket= 2000000
thread= 16
silent= 0
label_prefix= 'll'

# Train the classifier
print('Start Trainning...')
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
    min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
    thread=thread, silent=silent, label_prefix=label_prefix)

# Test the classifier
print('Start Test...')
result = classifier.test(test_file)
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)
```

代码部分很简单，输入的训练测试用的文件格式按照fasttext要求的格式：
	ll32 迅游 编程 老爷 输送 回到 中国农业银行 北京分行 信息安全 小爱 火绒 高速 安防 后勤 人才 安全 驾驶 超悦 永信 在线教育 守望 少儿 智能 微云 庄稼 益安 cn 悦茂 联赛 酒家 长亭 东升 web 哈勃 精神状态 千聊 中国青少年发展基金会 rainbow 小米 兰州 望远镜 水果 运维 转回 花童 美团 香里 姥姥家 平谷 果多美 北京 车车 开放 大巴 路况 春秋 石屋 网络安全 大侠 百行 客官 查导 榉树 合乘 吱车 婚姻登记 驾驶者 为青 vipjr itutorgroup 企业级 前沿技术 hubble netman 其子 普世化 pwnhub 应用层 telescope 护网杯 微油 电踏车 店查 尧西 origin 上助家 导及 桃叶谷 宇宙空间 及乐 火方 升空 黄栌 如履薄冰 乐乘 若隐若现 处室 此圈 看雪 控管 婚俗
	ll42 猪扒 同事 餐饮 九朝会 眼镜 区块 猎豹 数字 研学 有助于 平均 状况 管理者 沦陷 主义 食品市场 image 嘉宴 com 多点 快人 财经 振兴 德鲁克 傅盛 ceo 音米 玲香 men 版权 金刚 新生 笔记 图像 版块 复购 生产力 信文 不接 打卡 空间设计 一味 乡村 形象店 爱马 小腰 合生汇 望京 停车 代驾 颠簸 文旅 田园 货币 数千万 邦德 紫牛 识堂 华少 绵密 混沌 川菜 奶泡 

其中ll是标签的前缀，这是fasttext的格式要求，后边跟的是标签，在我们的场景下是用户的年龄。再之后接的是标签对应的文本数据，需要注意这里需要预先完成分词，去除停用词等文本预处理操作，然后分词后的文本用空格分隔。（fasttext针对的英文文本天然就是这种形式）

再之后就是设定一些参数，其中embedding的维度dim我们设置为30，文本分类往往不需要很长的embedding，因为不需要太过于依赖深层的语义。然后是学习率lr，epoch为迭代轮数。min_count体现在对于冷门词的处理，对于在数据集中出现的词频低于min_count的部分，fasttext将用UNK来代替以避免词表的爆炸。然后word_ngrams很重要，对于不需要利用上下文的话就让word_ngrams=1,这样运行效率最高。一般不会让word_ngrams>3否则运行效率会非常低，并且得到的模型会非常巨大。

fasttext做文本分类的时候也是有用到词向量的（而不是像之前说的简单的BOW），并且在模型保存的时候存储下来了：
 
fasttext是可以读取pre-train vec的。ft.supervised()加上参数pretrained_vectors，     pretrained word vectors (.vec file) for supervised learning []。
### keras的fasttext实现


### fasttext存在的问题

fasttext就非常敏感，原因在于它是将所有embedding直接avg后过NN的，噪声多了以后avg的结果自然带不了什么信息了，说实话是太粗暴了。
Fasttext怎么也对短文本有偏好呢？预测概率高的总是短文本，即使有些很不靠谱的。这个必须解决啊！
我猜想是否是因为它们处理变长文本的方式不同导致的，CNN会将所有变长文本padding到最长文本的长度，所以短文本的空白部分其实是有填充的。而Tgrocery和Fasttext直接构建一个BOW的模型(Fasttext的BOW是词向量的叠加，此外fasttext是不关心一个句子的长度的，无论多长都会用avg-pooling来处理它，因此不存在CNN遇到的一些padding的问题)，因此短文本的BOW就更纯净。简单说就是padding操作实际给短文本添加了白噪声，使得模型不会对短文本有偏。
	回头看这里又有两个截然不同的观点，后边我做keras-fasttext实验的时候发现简单的zero-padding有很多问题(这种padding引入的噪声非常大)，然而这里CNN就是用了zero-padding(keras.preprocessing.sequence.pad_sequences)才使得不会对短文本置信度有偏。并且既然keras提供这个库表明存在即是合理的。这两种观点我看起来都有道理，还是需要根据实际情况决定吧。
fasttext有明显的对短文本概率高的倾向了，因为fasttext是直接粗暴做avg-pooling后就softmax：
 


### mask的意义
在softmax的场景下，embedding加mask非常重要，如果不加mask短文本结果会非常差！终于验证了embedding mask的意义，以及fasttext为什么work well的原因：
 
model相比mdelC只是去掉了mask。
 
This can actually cause a fairly substantial fluctuation in performance in some networks. Suppose that instead of a convnet, we feed the embeddings to a deep averaging network. Then the varying number of nonzero pad vectors (according to which training batch the example is assigned in SGD) will very much affect the value of the average embedding. I've seen this cause a variation of up to 3% accuracy on text classification tasks. One possible remedy is to set the value of the embedding for the pad index to zero explicitly between each backprop step during training. This is computationally kind of wasteful (and also requires explicitly feeding the number of nonzero embeddings in each sample to the network), but it does the trick. I would like to see a feature like Torch's LookupTableMaskZero as well.
这里也提到实在不行就每次将padding emb强制设置为0，否则用mask是最好的。
	最后说下keras Embedding的mask_zero机制，它不会return [0,..0] vec for symbol 0，相反，Embedding layer的参数是不受影响继续训练的，mask_zero只是给了一个mask给后续的layer用，所以后续layer没有使用mask的话是会报错的。
 
经典的例子是后边接个LSTM：
 
### pooling策略对比

不管是去除低频词还是增加emb len，max-pooling的效果就是不如mean-pooling。难道这也是为什么fasttext用mean-pooling的原因？这里有个max-pooling很大的问题，每轮只能更新max的embedding，也就意味着每轮只有极少量的embedding能得到更新，这对于没有pre-train的wordVec是难以训练的。
	Stackoverflow上有个我想问的问题，tf.max/min能否对多个value同时计算梯度并更新？https://github.com/tensorflow/tensorflow/issues/16028。可能需要自定义gradient计算函数，因为tf会对所有定义好的op提供默认的梯度计算方法，比如：
 
可以看到明确说如果tf.max有多个same max值，那么会把梯度平均分给这几个value。
有两种方法改善tf.max的梯度影响范围，一种是自定义gradient计算函数，另一种是”softening” the max function，like the Lp-norm：

于是自己实现了一个Lp-norm-pooling：

这里超参数选p=2的就是最常见的L2-norm，发现当p比较大的时候无法收敛，怀疑是数值上溢，但是p=10就上溢也太夸张了吧。即使是L2-norm也很容易发散，应该不是好的pooling策略。
换一种方式验证，使用avg-pooling训练一版emb做为max-pooling的pre-train emb，效果明显好多了，并且不只是吃pre-train emb的老本，是可以在此基础上持续优化的。
突然有个好想法，在emb layer后边接一个dropout不就能够在使用max-pooling的情况下更好训练么？试了一下没什么效果，dropout的做法是让神经元的输出强制变0，所以从model.summary()看是没有shape的变化的：

但这还是不符合max-pooling的要求啊，0 output适合的是mean-pooling。当然dropout force to 0对于大部分NN是有意义的，因为神经网络本质是多层神经元乘加+非线性堆叠起来，0*w会一直为0，所以前期dropout的神经元在后边所有的网络都不会造成影响。
转念一想，我把Dropout放在InputLayer，而不是EmbeddingLayer后感觉就OK了。然而又有新问题，dropout后又有些新句子变成全0了导致训练又出现nan。于是又需要做一个condition操作了，注意在设计网络结构的时候不能直接将原始操作代码逻辑直接操作神经元，而是需要将逻辑封装在一个自定义层中的call()部分，就像上边自定义mask max-pooling layer，网络只能够由层组成。另外，tf提供了lambda layer可以只需添加一个lambda表达式作为逻辑，keras.layers.Lambda(lambda wide: wide**2)，这个对于大多数简单逻辑更方便，毕竟本质就只需要改写call()而已。于是一个condition_dropout的写法：
 input_cond_drop = keras.layers.Lambda(lambda x: K.switch(
            K.tf.count_nonzero(x)>5,Dropout(0.5)(x),x))(input)
	这个适用于对输入做条件dropout，太稀疏的输入就不dropout了。Keras默认dropout在predict阶段是不生效的，如果想要predict也dropout，可以定义一个permanent dropout: md.add(Lambda(lambda x: K.dropout(x, level=0.9)))。这个dropout是backend提供的感觉同keras Dropout Layer不太一样。

另外有人做了我类似想法的事情：
 

