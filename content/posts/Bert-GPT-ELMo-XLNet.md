---
title: Bert-GPT-ELMo-XLNet
date: 2022-09-06T15:10:05+08:00
categories:
- NLP
tags:
- NLP
katex: true
---

## Bidirectional Encoder Representation from Transformer(Bert)

### Bert的基本理解

Bert是一个能够适应各类NLP问题的通用预训练架构。其通过在大量的训练语料上以自监督学习的方式学习到词的特征表示，使得不同的NLP任务只需要对Bert进行微调，就能达到预期的效果，而不需要重新设计模型架构。

### Bert的预训练任务

BERT是一个多任务模型，是由两个自监督任务组成。

#### Masked Language Model(MLM)：

在MLM任务的训练过程中，输入语料中15%的词会被随机Mask掉，而在该语料多次参与训练的过程中，80%的时候被Mask的词会被替换成[Mask]，10%的时候会被替换成其他任意的词，10%的时候会保留原词。然后，模型根据上下文预测被Mask的词。

因为如果每次都把被Mask掉的词替换成[Mask]，则模型则可能不会学习到该次的特征表示，而随机替换成任意词，则是为了不让模型意识到[Mask]与原词的关系。

通过MLM，使得Bert模型能够更好地根据上下文预测当前词，并且赋予了模型一定程度的纠错能力。

#### Next Sentence Prediction(NSP)：

在NSP任务的训练过程中，输入语料为多对连续的句子，将50%的句子对中的一句换成其他任意的句子，使得连续的句子对和不连续的句子比例为1:1。然后，模型判断当前句子对是否连续。

通过NSP，使得Bert模型能够理解句子之间的关系。

### Bert的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/Bert_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Bert的结构</div>
</center>

上图为Bert的结构，由多个Transformer全连接层组成，因此Bert能够结合当前词上下文的语境进行预测。

Bert的输入是一个句子对的 Word Embedding + Position Encoding + Segment Encoding，在首句的句首加上一个特殊的Token[CLS]，在首句和尾句的句尾也加上一个特殊的Token[SEP]。其中，Segment Encoding只有两种状态，用来标记该词属于首句还是尾句。

### Bert的Fine-Tuning

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/Bert_Fine-Tuning.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fine-Tuning的四种形式</div>
</center>

上图为对Bert模型进行Fine-Tuning的四种主要形式，对应四种主要的NLP任务。

1. 对于普通的分类任务，输入是一个序列，所有的Token都是属于同一个Segment，即Segment Encoding相同，可以对[CLS]的最后一层输出使用分类器进行分类。

2. 对于相似度计算的任务，输入是两个序列，只需要对[CLS]的最后一层输出使用分类器进行分类。

3. 对于问答的任务，输入是问题序列和一段包含答案的paragraph序列，输出为答案在paragraph序列中的开始和结束位置。

4. 对于序列标注的任务，比如命名实体识别，输入是一个序列，需要对除了[CLS]和[SEP]之外的每个时刻的最后一层输出进行分类。

### 问题

1. MLM任务使得Bert能够借助上下文理解语义，但同时导致其预训练的数据与微调的数据不匹配，因此其适合处理自然语义理解类任务，不适合自然语言生成式任务。

2. 由于Bert模型要求序列的长度必须一致，如果过长则需要截断，过短则用padding补齐，因此其适合句子和段落级别的任务，不适用于文档级别的任务。

## Generative Pre-training Transformer(GPT)

### GPT的基本理解

GPT的思想与Bert类似，也是通过无标签的数据上训练得到一个通用的语言模型，然后再根据特定的任务进行微调。其采用多层单向Transformer作为特征抽取器。

### GPT的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/GPT_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">GPT的结构</div>
</center>

上图为GPT的结构，与Bert类似，但是采用的是单向Transformer，即训练时通过Mask Multi-Head Attention模块遮蔽Ground Truth中当前时刻之后的词。输入为当前词的Word Embedding + Position Encoding。

与Bert类似，GPT在完成预训练之后，不同的NLP任务只需要对其进行微调即可。

GPT作为单向语言模型，与Bert不同，无法同时捕捉上下文信息，因此更适合自然语言生成式任务，不适合自然语言理解类任务。

### GPT-2和GPT-3

| 模型 | 参数数量 | 预训练数据量 |
| :---: | :---: | :---: |
| GPT | 1.17亿 | 约5GB |
| GPT-2 | 15 亿 | 40GB |
| GPT-3 | 1750 亿 | 45TB |

GPT-2和GPT-3的目标是进一步提高GPT模型的泛化能力。GPT-2和GPT-3相比GPT-1，没有对模型结构进行过多的创新与设计，而是使用了更多的参数和更大的数据集。GPT-2和GPT-3验证了通过增加参数数量和预训练数据量，得到模型具有更高的泛化能力，可以在仅给定任务说明和少量示例的情况下可以迁移到其他NLP任务中，不需要额外的Fine-Tuning样本进行有监督训练。

## Embedding from Language Model(ELMo)

### ELMo的基本理解

由于传统的Word Embedding都是固定的，即每个词有唯一的Word Embedding，因此不能很好地解决一词多义的情况。ELMo利用了多层双向的LSTM结构，其中低层的LSTM用于捕捉比较简单的语法信息，高层的LSTM捕捉依赖于上下文的语义信息，根据当前上下文对Word Embedding进行动态调整。

### ELMo的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/ELMo_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">ELMo的结构</div>
</center>

上图为ELMo的结构，输入为当前词的Word Embedding，主体部分由一个从左到右的多层LSTM和一个从右往左的多层LSTM构成，前者用于捕捉当前词与前文的关系，后者用于捕捉与后文的关系。最后将两者的输出与当前词的Word Embedding加权求和后作为当前词根据上下文调整后的Contextual Embedding。

虽然ELMo采用了两个多层LSTM，一个用于捕捉上文信息，一个用于捕捉下文信息，但是由于RNN的顺序结构，仍然无法做到同时捕捉上下文信息，因此仍为单向语言模型，与GPT相同，更适合自然语言生成式任务，不适合自然语言理解类任务。同时，ELMo所使用的LSTM相对于Transformer特征捕捉的能力也较弱。

## XLNet

### XLNet的基本理解

由于以Bert为代表的autoencoding(AE)language model虽然可以捕捉上下语言特征，但是由于Fine-Tuning的数据中没有MASK，使得与预训练数据不一致，容易引入误差。同时，以GPT为代表的autoregressive(AR)language model为单向语言模型，无法同时捕捉上下文信息。

为了兼顾AR的方法可以更好地学习单词之间依赖关系的优点，以及AE的方法可以更好地利用深层的双向信息的优点，XLNet使用了Permutation Language Model(PLM)的方法。

XLNet将句子中的单词随机排列，然后采用AR的方式预测末尾的几个单词，这使得在预测单词的时候就可以同时利用该单词双向的信息，并且能学到单词间的依赖。

### XLNet的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/XLNet_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">XLNet的结构</div>
</center>

上图为XLNet的结构。XLNet的输入为句子的 Word Embedding 和 Position Encoding，采用了Two-Stream Self-Attention机制，通过Attention掩码机制模拟句子中单词的随机排列，即仅向下层传递打乱后该单词位置之前的单词信息，其中Content Stream与传统的transformer attention机制相同，向下层传递该单词及打乱后其位置之前的单词内容信息，而Query Stream则向下层传递该单词的原始位置信息以及打乱后其位置之前的单词内容信息。

XLNet还采用了Segment Recurrence Mechanism(段循环)机制，即将上一个句子输出的信息保存下来，用于当前句子的计算，使模型可以拥有更广阔的上下文信息。同时，为了避免引入上一个句子的信息后所导致不同单词拥有相同的Positional Encoding的问题，XLNet采用了Relative Positional Encoding(相对位置编码)，即以单词之间的相对位置作为Position Encoding。

## Reference

https://zhuanlan.zhihu.com/p/48612853
https://wmathor.com/index.php/archives/1456/
https://www.cnblogs.com/gczr/p/11785930.html
http://fancyerii.github.io/2019/03/09/bert-theory/
https://www.cnblogs.com/sandwichnlp/p/11947627.html
https://zhuanlan.zhihu.com/p/350017443
https://zhuanlan.zhihu.com/p/200978538
https://zhuanlan.zhihu.com/p/72309137
https://zhuanlan.zhihu.com/p/63115885
https://www.jianshu.com/p/2b5b368cbaa0
