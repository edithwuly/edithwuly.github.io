---
title: RNN-LSTM
date: 2022-09-06T15:10:05+08:00
categories:
- NLP
tags:
- NLP
katex: true
---

## Recurrent Neural Network(RNN)

### RNN的基础理解

对于RNN而言，一个序列当前的输出与之前的输出也有关。具体的表现形式为网络会对前面的信息进行记忆，保存在网络的内部状态中，并应用于当前输出的计算中，即隐含层之间的节点不再无连接而是有链接的，并且隐含层的输入不仅包含输入层的输出还包含上一时刻隐含层的输出。如果想要通过RNN获得相同的输出，则不仅需要保证输入相同，也需要保证输入的序列相同。

### RNN的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/standard_RNN.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">RNN标准结构</div>
</center>

上图为RNN的标准结构，其中W、V、U为权重，o为输出，y为标记，x为输入，h为隐藏层状态。L为输入o与y的损失函数。可以看出，t时刻的输出o<sup>(t)</sup>，不仅仅取决于t时刻的输入x<sup>(t)</sup>，还取决于t-1时刻的隐藏层状态h<sup>(t-1)</sup>，即

$$h^{(t)} = \phi (Ux^{(t)} + Wh^{(t-1)} + b)$$
$$o^{(t)} = Vh^{(t)} + c$$

其中$\phi$为激活函数，一般使用tanh函数或者sigmoid函数。由于tanh函数是零点中心对称，相比sigmoid函数可以收敛的更好。

BPTT(back-propagation through time)是RNN常用的训练方法，该方法基于时间反向传播，沿着W、V、U三个权重矩阵的负梯度方向不断优化，直至收敛，即损失函数L达到最小值。

原始的RNN要求输入与输出一一对应，但是这样的结构并不能解决所有问题，在实际使问题中，存在大量输入与输出序列不等长的问题，例如多输入单输出（分类问题），或者N输入M输出（翻译问题）等等。因此，Seq2Seq模型（Encoder-Decoder模型）出现了。

### Seq2Seq模型

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/Seq2Seq_model.jfif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Seq2Seq模型结构</div>
</center>

上图为Seq2Seq模型基本结构，先使用一个RNN对输入进行编码，得到中间结果，再使用一个RNN对中间结果进行解码，得到输出。中间结果大多为Encoder的最后一个隐藏层状态。

### 存在问题

在RNN的反向传播过程中，由于W和U的偏导的求解需要涉及到历史数据，假设只有三个时刻，那么在第三个时刻L对W的偏导数为：
$$\frac{\partial L^{(3)}}{\partial W} = \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial W} + \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial h^{(2)}}\frac{\partial h^{(2)}}{\partial W} + \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial h^{(2)}}\frac{\partial h^{(2)}}{\partial h^{(1)}}\frac{\partial h^{(1)}}{\partial W}$$
L对U的偏导数为：
$$\frac{\partial L^{(3)}}{\partial U} = \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial U} + \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial h^{(2)}}\frac{\partial h^{(2)}}{\partial U} + \frac{\partial L^{(3)}}{\partial o^{(3)}}\frac{\partial o^{(3)}}{\partial h^{(3)}}\frac{\partial h^{(3)}}{\partial h^{(2)}}\frac{\partial h^{(2)}}{\partial h^{(1)}}\frac{\partial h^{(1)}}{\partial U}$$
类推到t时刻，则
$$\frac{\partial L^{(t)}}{\partial W} =  \sum^t_{k=0}\frac{\partial L^{(t)}}{\partial o^{(t)}}\frac{\partial o^{(t)}}{\partial h^{(t)}}(\prod^t_{j=k+1}\frac{\partial h^{(j)}}{\partial h^{(j-1)}})\frac{\partial h^{(k)}}{\partial W}$$
$$\frac{\partial L^{(t)}}{\partial U} =  \sum^t_{k=0}\frac{\partial L^{(t)}}{\partial o^{(t)}}\frac{\partial o^{(t)}}{\partial h^{(t)}}(\prod^t_{j=k+1}\frac{\partial h^{(j)}}{\partial h^{(j-1)}})\frac{\partial h^{(k)}}{\partial U}$$
由于
$$h^{(t)} = \phi (Ux^{(t)} + Wh^{(t-1)} + b)$$
所以
$$\frac{\partial h^{(j)}}{\partial h^{(j-1)}} = \phi'W$$

1. 梯度消失
    如果取sigmoid函数作为激活函数的话，sigmoid函数导数值范围为(0,0.25]，那么必然是一堆小数在做乘法，结果就是越乘越小。随着时间序列的不断深入，小数的累乘就会导致梯度越来越小直到接近于0。也就是说，梯度被近距离梯度主导，导致难以学到远距离的依赖关系。

    假设需要预测"I grew up in France... I speak fluent French"的最后一个词，那需要保留第5个词"France"的信息，但是"France"与当前预测位置之间的间隔就相当大，以至于RNN难以学习到当前预测位置与"France"之间的依赖关系。

    如果取tanh函数作为激活函数，tanh函数的导数范围是(0,1]，相对好一些，但是也没有解决根本问题。

2. 梯度爆炸

    如果选用的激活函数的导数值比较大或者W非常大的时候，则随着时间序列的不断深入，累乘的结果会接近于无穷。

    处理梯度爆炸可以采用梯度截断的方法，将梯度值超过阈值的梯度手动降到预设值。虽然梯度截断会一定程度上改变梯度的方向，但梯度截断的方向依旧是朝向损失函数减小的方向。

## Long Short Term Memory(LSTM)

### LSTM的基础理解

LSTM在传统RNN的基础上引入了门（gate）机制用于控制特征的流通和损失。

### LSTM的结构

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/LSTM_structure.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">LSTM模型结构</div>
</center>

上图为LSTM的基本结构。其中，遗忘门$f_t$用来决定上一个时刻的单元状态$C_{t-1}$的哪些特征被用于计算当前时刻的单元状态$C_t$。单元状态更新值$\tilde{C_t}$由当前输入$x_t$和上一个时刻的隐层状态$h_{t-1}$决定,输入门$i_t$用来决定$\tilde{C_t}$的哪些特征被用于计算当前时刻的单元状态$C_t$。最后，输出门$o_t$用来计算当前的隐层状态。

通过遗忘门的控制，就可以决定在长文本的学习中，哪些特征需要被保留，例如在"I grew up in France... I speak fluent French"的例子中，"France"的信息就可以被保留用于预测最后一个词。

### 梯度消失和梯度爆炸

1. 梯度消失

    LSTM刚提出时单元状态的更新函数为
    $$C_t = C_{t-1} + i_t \times \tilde{C_t} $$
    相当于遗忘门$f_t=1$，从而$C_{t-1}$的梯度可以直接传给$C_t$，不会消失。但是在其他传播路径上，LSTM的梯度流和普通RNN没有太大区别，依然会消失。由于总梯度 = 各传播路径的梯度之和，即便其他传播路径梯度消失了，只要保证有一条传播路径梯度不消失，总梯度就不会消失。因此LSTM通过改善单元状态更新路径上的梯度问题拯救了总梯度。

    对于带遗忘门的LSTM来说，如果遗忘门$f_t \approx 1$，则类似与原始的LSTM。如果遗忘门$f_t \approx 0$，例如情感分析任务中有一条样本 “A，但是 B”，模型读到“但是”后选择把遗忘门设置成 0，遗忘掉内容 A，但这时模型是故意阻断梯度流的。最后，当$f_t\in [0, 1]$时，在这种情况下只能说LSTM改善了梯度消失的状况。

2. 梯度爆炸

    因为总梯度 = 各传播路径的远距离梯度之和，任意一条传播路径的梯度爆炸，总梯度都会爆炸，因此LSTM仍无法避免梯度爆炸的问题。不过，由于LSTM改善单元状态更新路径上的梯度比较稳定，其他传播路径和原始RNN相比多又经过了很多次激活函数，而激活函数的导数大都小于1，因此LSTM发生梯度爆炸的频率要低得多。而就算发生梯度爆炸也可以通过梯度截断来解决。

## GRU

### GRU的基础理解

与LSTM相比，GRU去除掉了细胞状态，使用隐藏状态来进行特征的传递。它只包含两个门：更新门和重置门。GRU相对于LSTM而言，参数更少，因而训练稍快或需要更少的数据来泛化。另一方面，如果你有足够的数据，LSTM的强大表达能力可能会产生更好的结果。

### GRU的结构
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="/pic/GRU_structure.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">GRU模型结构</div>
</center>

上图是GRU的基本结构。其中，重置门$r_t$用来控制当前的候选隐藏层状态中需要保留多少之前的记忆，比如如果$r_t$为0，那么当前的候选隐藏层状态$\tilde{h_t}$只包含当前词的信息。更新门$z_t$用于控制当前的隐藏层状态中，来自于前一时刻的隐藏层状态$h_{t-1}$的特征信息，与来自于当前时刻的候选隐藏层状态$\tilde{h_t}$的特征信息的比例。

一般来说那些具有短距离依赖的单元重置门$r_t$比较活跃，因为如果 $r_t$为1，而$z_t$为0，那么相当于变成了一个标准的RNN，具有较好的短距离依赖，具有长距离依赖的单元$z_t$比较活跃。

## Reference

https://blog.csdn.net/qq_16234613/article/details/79476763
https://blog.csdn.net/weixin_41089007/article/details/96474760
https://blog.csdn.net/zhaojc1995/article/details/80572098
https://zhuanlan.zhihu.com/p/28687529
https://zhuanlan.zhihu.com/p/42717426
https://www.jianshu.com/p/9dc9f41f0b29
https://www.zhihu.com/question/34878706/answer/665429718
https://blog.csdn.net/lreaderl/article/details/78022724
https://zhuanlan.zhihu.com/p/138267466
