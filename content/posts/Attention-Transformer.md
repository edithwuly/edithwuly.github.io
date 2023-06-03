---
title: Attention-Transformer
date: 2022-09-06T15:10:05+08:00
categories:
- NLP
tags:
- NLP
katex: true
---

## Attention

### Attention的基本理解

对于没有采用Attention的模型而言，每个输入对于输出的影响力都是一样的。以传统的采用了Seq2Seq模型的机器翻译为例，假设需要翻译"Tom chase Jerry"，由于Encoder会先把输入转换为中间结果，在这一步中，输入的每个单词都具有相同的影响力，然后再由Decoder把中间结果解析成“汤姆追逐杰瑞”。

对于较短的输入而言，传统的RNN模型能够较完整的保存输入的信息，但是随着输入的加长，如果把所有输入等权重地压缩中一个中间结果，由于RNN模型本身存在远距离依赖消失的问题，必然会丢失掉部分输入的特征信息，从而导致最后的输出不能完整地反映输入。

在加入了Attention之后，能够细化每个输入对于输出的影响力大小，通过对输入的特征信息加以相应的Attention权重，使得输出能够更好的反映高影响力的信息特征。例如对于"Tom chase Jerry"的翻译，Encoder可能产生三个中间结果，其中"Tom"对于第一个中间结果的影响力更高，"chase"对于第二个中间结果的影响力更高，"Jerry"对于第三个中间结果的影响力更高。

### Attention的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/Attention_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Attention的计算过程</div>
</center>

上图为大多数Attention的计算过程。在第一阶段，根据当前的Query向量，计算其与各个Key向量的相似性或者相关性，大多使用向量点积。在第二阶段，使用SoftMax()函数进行归一化，将第一阶段产生的计算分值整理成权重之和为1的概率分布，得到的结果就是各个输入对于当前输出的Attention权重。在第三阶段，将各个输入的Value向量采用Attention权重进行加权求和，得到当前输出的中间结果。

例如对于传统的Seq2Seq模型来说，在t时刻的Query向量可以采用t-1时刻Decoder的隐藏层状态$h_{t-1}$，Key向量可以采用Encoder各个输入的隐藏层状态$h_i$，Value向量则是各个输入向量$x_i$。

### Self-Attention

在一般的Seq2Seq模型中，输入和输出的内容是不一样的，比如对于英中机器翻译来说，输入是英文句子，输出是中文句子，Attention机制发生在各个输出的Query向量和所有输入之间。而Self-Attention，指的不是输入和输出之间的Attention机制，而是输入与输入之间或者输出与输出之间的Attention机制。

在上一节的计算过程中，每个输入的Query向量、Key向量和Value向量可以通过自定义$W_q,W_k,W_v$矩阵对其字向量进行线性变换得到。

通过采用Self-Attention，可以直接计算出句子中任意两个单词的相关性，使得更容易捕获句子中长距离的相互依赖的特征，例如短语结构或者指代结构。

### Multi-Head Attention

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/multi-head_attention_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Multi-Head Attention的计算过程</div>
</center>

上图为Multi-Head Attention的计算过程。相比于普通的Self-Attention，Multi-Head Attention采用了多组线性变化矩阵$W_q,W_k,W_v$，生成的多个Attention权重矩阵。

Multi-Head Attention使得能够同时关注一个单词与其他单词不同的角度的相关性，例如一个代词同时也是句法结构的一部分。

## Transformer

### Transformer的基本理解

由于在RNN模型中，t时刻的输出依赖于t-1时刻的隐层状态，即只能根据输入的顺序依次进行计算，极大地限制了模型的并行能力。同时，RNN模型还存在难以保存远距离依赖关系的问题。

对此，Transformer选择抛弃传统的CNN和RNN结构，由且仅由Self-Attenion和Feed Forward Neural Network组成，将序列中的任意两个位置之间的距离是缩小为一个常量，而且由于不是顺序结构，因此具有更好的并行性，符合现有的GPU框架。

### Transformer的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/Transformer_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Transformer的结构</div>
</center>

上图为Transformer的结构。

#### Encoder部分：

Encoder部分的输入每个字的Word Embedding + Position Encoding，由于Transformer模型采用的非顺序结构，所以必须额外提供每个输入的位置信息，才能识别出语言中的顺序关系。

Encoder的主体部分为多个Encoder的叠加，每个Encoder由Self-Attention模块和Feed Forward Neural Network模块组成，Self-Attention模块用来捕获句子中长距离的相互依赖的特征，由于Transformer中的Self-Attention采用的是Multi-Head Attention机制，因此，对于一个字会产生多个Attention权重矩阵，每个字的value向量与不同的Attention权重矩阵加权求和后，会产生多个加权矩阵。而Feed Forward Neural Network模块则会将多个加权矩阵连接在一起，拼成一个大矩阵，用激活函数激活后，线性映射成所需要的大小。

对于每一个Self-Attention模块和Feed Forward Neural Network模块的结果，都需要Add & Normalize层。首先，进行Residual Connection，将该模块的输入与输出进行相加，使得在反向传播过程中，该模块求偏导的时候多一个常数项，一定程度上可以缓解梯度消失问题。然后，进行Normalize，把该模块的输出转化成均值为0方差为1的形式，加快收敛的速度。

#### Decoder部分：

在推断时，Decoder部分的输入为之前输出的Word Embedding + Position Encoding。但是，在训练时，如果同样采用之前的输出作为输入，可能会因为偏差的累积，导致预测的结果越来越差，因此，在训练时，Decoder部分的输入为Ground Truth的Word Embedding + Position Encoding。

但是，由于Decoder需要按照循序依次进行decode，在预测t时刻的输出时，只能看到t-1时刻以及之前的输出，因此，在训练时，Decoder的Self-Attention模块需要对Ground Truth中t时刻以及之后的信息进行Mask。具体操作为，计算Attention权重矩阵的时候，在进行归一化之前，保持t-1时刻以及之前的向量点积不变，将t时刻以及之后的向量点积变为$-inf$，这样能够使得t时刻以及之后的Attention权重无限趋近于0。

Decoder中的Encoder-Decoder Attention模块使用Attenton机制对Encoder的输出进行decode，其Query向量为Decoder中Self-Attention模块的输出，Key向量和Value向量由Encoder的输出与$W'_k$和$W'_v$相乘，进行矩阵变换得到。

Decoder中的Feed Forward Neural Network模块则与Encoder相同，由于Decoder的Self-Attention和Encoder-Decoder Attention模块都采用了Multi-Head Attention的形式，因此会生成多个Attention权重矩阵，由Feed Forward Neural Network模块进行拼接，激活，并线性映射到需要的大小。

最后，Decoder部分将输出向量线性映射成与词典大小相同的向量，并Softmax函数激活，则可以得到词典中每个词的概率。

### 存在问题

1. Transformer局部特征的捕捉能力不如RNN强

2. Transformer丢失了输入的位置信息，虽然使用了Position Encoding进行弥补，但是不能保证Position Encoding在经过多次线性变换后，仍然完整地保留了位置信息。

## Reference

https://blog.csdn.net/malefactor/article/details/78767781
https://wmathor.com/index.php/archives/1438/
http://mantchs.com/2019/09/26/NLP/Transformer/
https://zhuanlan.zhihu.com/p/48508221
http://jalammar.github.io/illustrated-transformer/
http://www.qishunwang.net/news_show_87606.aspx
https://zhuanlan.zhihu.com/p/330483336
