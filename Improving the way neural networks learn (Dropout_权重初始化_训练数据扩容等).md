##regularization
防止过拟合

###L2 regularization

###L1 regularization
L1 regularization shrinks the weight much more than L2 regularization. The net result is that L1 regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero

###Max-norm regularization

###Batch Normalization

代码参考：[implementing-batch-normalization-in-tensorflow](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

- Normalization作用时机：$x_{hidden}=x_{input}w+b$之后，activation之前
- Normalization作用对象：以隐藏层中的每一个单独神经元为单位，normalize该神经元的输入数据集合

假设隐藏层输入为$x=(x^{(1)},\cdots,x^{(n)})$，则对每一个输入维度$x^{(k)}$进行normalize：
$$
\hat{x}^{(k)}={ {x^{(k)}}-E[x^{(k)}] \over \sqrt{Var[x^{(k)}]} }
$$

为了保证原网络表达性，
$$
y^{(k)}=\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$
当$y^{(k)}=\sqrt{Var[x^{(k)}]}，\beta^{(k)}=E[x^{(k)}]$时就变回原始输入值了

####一些性质
Batch Normalization also makes training more resilient to the parameter scale

With Batch Normalization, back-propagation through a layer is unaffected by the scale of its parameters. The scale does not affect the layer Jacobian nor, consequently, the gradient propagation. Moreover, larger weights lead to smaller gradients, and Batch Normalization will stabilize the parameter growth.

$$
\begin{aligned}
BN(Wu)&=BN((aW)u) \\
\\
{\partial BN((aW)u) \over \partial u}&={\partial BN(Wu) \over \partial u} \\
{\partial BN((aW)u) \over \partial (aW)}&={1\over a} \cdot {\partial BN(Wu) \over \partial W}
\end{aligned}
$$

<font color=red>jfo的实验现象</font>：当batch normalization所在隐藏层神经元数目较多时，check_gradient过程计算出的梯度与真实梯度之差$diff$为1e-5 ~ 1e-6数量级，而关闭batch normalization后$diff$为1e-11 ~ 1e-12数量级，说明batch normalization能让权重参数的梯度变化率更大。且隐藏层神经元越多，差距越大。实验代码如下所示：

```python
# 代码分支batchnorm_0.1
def drawMyMLP_moon():
    from sklearn import datasets, linear_model

    x, y = datasets.make_moons(10, noise=0.20)
    print x.shape, y.shape

    model = my_mlp.MLP([my_mlp.Layer(1000, activation='sigmoid', batch_norm=False)],
                       learning_rate=0.05, n_iter=20000, Lambda=1e-4, verify=True)
    model.fit(x, y)
```

*具体算法参考论文《Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift》*

<img src=img/batchnorm_alg1.png width=400>

<img src=img/batchnorm_gradient.png width=400>

<font color=red>注意：上面最后两项求导错误，应当如下所示：</font>
$$
\begin{aligned}
{\partial \ell \over \partial \gamma}&=\color{red}{1\over m}\sum_{i=1}^m {\partial \ell \over \partial y_i} \hat{x}_i \\
{\partial \ell \over \partial \beta}&=\color{red}{1\over m} \sum_{i=1}^m {\partial \ell \over \partial y_i} = {\partial \ell \over \partial y_i}
\end{aligned}
$$

<img src=img/batchnorm_alg2.png width=400>

Algorithm2中第10步的$\mu_{\mathcal{B}}$是每一个mini-batch $\mathcal{B}$的期望值，$E[x]\leftarrow E_{\mathcal{B}}[\mu_{\mathcal{B}}]$对所有$\mu_{\mathcal{B}}$求平均；同样，$Var[x]\leftarrow {m\over m-1}E_{\mathcal{B}}[\sigma^2_{\mathcal{B}}]$对所有mini-batch的方差作无偏差发差估计(unbiased variance estimate)


Below is the distribution over time of the inputs to the sigmoid activation function of the first five neurons in the network’s second layer

参考[“implementing-batch-normalization-in-tensorflow”](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

```python
fig, axes = plt.subplots(5, 2, figsize=(6,12))
fig.tight_layout()

for i, ax in enumerate(axes):
    ax[0].set_title("Without BN")
    ax[1].set_title("With BN")
    ax[0].plot(zs[:,i])
    ax[1].plot(BNs[:,i])
```

<img src=img/batchnorm_output_fig.png />

###Dropout

Dropout can be interpreted as a way of regularizing a neural network by adding noise to its hidden units.
We found that as a side-effect of doing dropout, the activations of the hidden units become sparse.

[*Dropout explanation*](http://neuralnetworksanddeeplearning.com/chap3.html#other_techniques_for_regularization): 
A related heuristic explanation for dropout is given in one of the earliest papers to use the technique*: "This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons." In other words, if we think of our network as a model which is making predictions, then we can think of dropout as a way of making sure that the model is robust to the loss of any individual piece of evidence. In this, it's somewhat similar to L1 and L2 regularization, which tend to reduce weights, and thus make the network more robust to losing any individual connection in the network.

[论文 A Simple Way to Prevent Neural Networks from Overfitting]():
Although dropout alone gives significant improvements, using dropout along with max-norm regularization, large decaying learning rates and high momentum provides a significant boost over just using dropout. <font color=red>A possible justification is that constraining weight vectors to lie inside a ball of fixed radius makes it possible to use a huge learning rate without the possibility of weights blowing up. The noise provided by dropout then allows the optimization process to explore different regions of the weight space that would have otherwise been difficult to reach.</font> As the learning rate decays, the optimization takes shorter steps, thereby doing less exploration and eventually settles into a minimum.

In a standard neural network, the derivative received by each parameter tells it how it should change so the final loss function is reduced, given what all other units are doing. Therefore, <font color=red>units may change in a way that they fix up the mistakes of the other units. This may lead to complex co-adaptations.</font> This in turn leads to overfitting because these co-adaptations do not generalize to unseen data. We hypothesize that for each hidden unit, dropout prevents co-adaptation by making the presence of other hidden units unreliable. Therefore, <font color=red>a hidden unit cannot rely on other specific units to correct its mistakes. It must perform well in a wide variety of different contexts provided by the other hidden units.</font>

下图为普通神经网络训练出的特征与dropout网络训练出的特征对比，可以看到dropout网络更强调单个神经元特征。(a) co-adapted in order to produce good reconstructions. Each hidden unit on its own does not seem to be detecting a meaningful feature. (b) the hidden units seem to detect edges, strokes and spots in different parts of the image

<img src=img/features_contrast.png width=540>

###DropConnect

Dropout与DropConnect对比：

<img src=img/Dropout_vs_DropConnect.png width=500>


[*论文DropConnect*算法](http://cs.nyu.edu/~wanli/dropc/dropc.pdf)

<img src=img/DropConnect_SDG_Training.png width=400>

<img src=img/DropConnect_Inference.png width=400>

具体代码实现可以参考[《Dropout and Dropconnect Code.pdf》](http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/Dropout.php)（采样求和稍微有点问题，应先激活后平均）

另外一个matlab代码实现参考[github DropConnect代码](https://github.com/billderose/DropConnect/blob/master/NN/nnff.m)，nnff.m实现了正确采样求值

###DropConnect预测(Inference)
*如下内容参考《DropConnect - Regularization of Neural Networks using DropConnect》3.2.Inference，<font color=red>下面W和v正好是论文中的W和v的转置</font>*

预测时，预测值$r$为：
$$
r = {1\over \mid M\mid} \sum_Ma(v(W \star M))
$$
$v$为上一层神经网络输出（当前层输入），$M$为masked weight matrix，$a$为激活函数，$\star$表示$M$和$W$简单按元素相乘（element-wise multiply）

为得到$r$需要计算$2^{\mid M\mid}$次不同mask $M$下的值，遍历$M$的所有可能取值，累加后取平均，这种方式不现实，换一种实现方式

单独看某个隐藏层的第$i$个神经单元的输入$u_i$：
$$
u_i = \sum_j v_{j}(W_{ji} M_{ji}) = \sum_j(v_{j}W_{ji})M_{ji}
$$

$v_{j}$是输入数据$V_{m\times n}$中的某一行第$j$列，$u_i$ is <font color=red>a weighted sum of Bernoulli variables $M_{ji}$($i$固定)</font>，can be approximated by a Gaussian via moment matching. 

The Central Limit Theorem lets us approximate a $Binomial(n,p)$ distribution with a $\mathcal{N}(np,np(1−p))$, so that instead of averaging over all possible networks, we compute the layer response as test time by sampling or numeric integration

$$
\begin{aligned}
E[u_i] &= E\left[\sum_j(v_{j}W_{ji})M_{ji}\right] \\
&= \sum_j E\left[v_{j}W_{ji} M_{ji} \right] \\
&=\sum_j v_{j}W_{ji} E[M_{ji}] \\
&=\sum_j v_{j}W_{ji} p \\
&= vW_{*i} p
\end{aligned}
$$

$$
\begin{aligned}
Var(u_i) &= Var\left(\sum_j(v_{j}W_{ji})M_{ji}\right) \\
&=\sum_j Var(v_{j}W_{ji}M_{ji}) \\
&=\sum_j v_{j}^2W_{ji}^2 Var(M_{ji}) \\
&=p(1-p)\sum_j v_{j}^2W_{ji}^2 \\
&=p(1-p) v^2W_{*i}^2 \\
&=p(1-p) (v\star v) (W_{*i} \star W_{*i})
\end{aligned}
$$

为求$u_i$，可以在$\mathcal{N}(np,np(1−p))$上采样k次，得$z_1,z_2,\cdots,z_k$，然后取平均值，正如`Algorithm 2 Inference with DropConnect`所示
$$
u_i={1\over k}\sum_{i=1}^k z_i
$$

####先平均后激活vs先激活后平均
$$
\sum_M a(v(W \star M)) \approx a\left(\sum_M v(W \star M) \right)
$$
严格来说这种近似有些问题，特别是对relu activation，论文中也有提到：
$设u \sim \mathcal(0,1)，a(u)=max(u,0)$，两者值有较大差距
$$
a(E_M[u])=0 \\
E_M[a(u)]={1 \over \sqrt{2} \pi} \approx 0.4
$$

只有当激活函数关于点$(E_M[u],a(E_M[u]))$对称时，左右两侧值能互补，上述近似才成立


<br>
>伯努利试验(Bernoulli experiment)是在同样的条件下重复地、相互独立地进行的一种随机试验。其特点是该随机试验只有两种可能结果：发生或者不发生。然后我们假设该项试验独立重复地进行了n次，那么我们就称这一系列重复独立的随机试验为n重伯努利试验

<div/>
>
$Binomial(n,p)$，二项分布（Binomial Distribution），即重复n次的伯努利试验，每次事件发生的概率是p，不发生的概率是1-p，N次独立重复试验中发生k次的概率是
$$
P(X=k)=C_n^k p^k (1-p)^{n-k}
$$
那么就说这个属于二项分布，二项分布有
$$
E[X]=np \\
Var(X)=np(1-p)
$$
>
证明：由二项式分布的定义知，随机变量X是n重伯努利实验中事件A发生的次数，且在每次试验中A发生的概率为p。因此，可以将二项式分布分解成n个相互独立且以p为参数的（0-1）分布随机变量之和
>
设随机变量$X(k)(k=1,2,3 \cdots n)$服从(0-1)分布，则$X=X(1)+X(2)+X(3)+\cdots +X(n)$，$X(k)$相互独立
$$
E[X]=E[X(1)+X(2)+X(3)+\cdots+X(n)]=np
$$
$$
\begin{aligned}
Var(X_i)&=E[X_i^2]-E[X_i]^2=p*1^2+(1-p)*0^2-E[X_i]^2=p(1-p) \\
Var(X_i)&=p*(1-E[X_i])^2+(1-p)(0-E[X_i])^2=p(1-p)
\end{aligned}
$$
$$
\begin{aligned}
Var(X)&= Var \left(\sum_i X_i \right) \\
&=\sum_{i,j} Cov(X_i,X_j) \\
&=\sum_i Var(X_i) + \sum_{i\ne j}Cov(X_i,X_j) &X_i与X_j不相关 \\
&=\sum_i Var(X_i) \\
&=np(1-p)
\end{aligned}
$$

<div/>
>
设随机变量序列$X_1,X_2,\cdots,X_n$互相独立，具有相同的数学期望$u$和方差$\sigma^2$，令$Y_n=\sum_{i=1}^n X_i$：
$$
Z_n={Y_n-E[Y_n] \over \sqrt{Var(Y_n)}} = {Y_n-nu \over \sqrt{n} \sigma} = {{Y_n \over n}-u \over {\sigma \over \sqrt{n}}} \longrightarrow \mathcal{N}(0,1)
$$
**中心极限定理**：设从均值为$u$、方差为$\sigma^2$，（有限）的任意一个总体中抽取样本量为n的样本，当n充分大时，样本均值的抽样分布近似服从均值为$u$、方差为${\sigma^2 \over n}$的正态分布
>
原来的分布不一定要符合正态分布，可以是任何的分布，可以是离散也可以是连续，即无要求。n为sample size，每次取n个样本，每次样本的mean分别为：$\overline{X_1}、\overline{X_2}、\overline{X_3} \cdots$，n越大这些样本均值越接近正态分布
>
- $n=1$：就如同原来的分布
- $n\to\infty$：mean为$u$方差为0，一条直线
>
<img src=img/Central_Limit_Theorem.png width=400>
>
**棣莫佛－拉普拉斯定理（服从二项分布的随机变量序列的中心极限定理）**：二项分布$Binomial(n, p)$以$\mathcal{N}(np, np(1-p))$为极限
>
<img src=img/binomial_normal.png width=400>


##better method for initializing the weight
Initialize each neuron’s weight vector as
$$w = {np.random.randn(n) \over sqrt(n)}$$
$randn$ samples from a zero mean, unit standard deviation gaussian, $n$ is the number of its inputs($n$是输入向量的特征维度，不是训练数据数量)

It turns out that we can normalize the variance of each neuron’s output to 1 by scaling its weight vector by the square root of its fan-in (i.e. its number of inputs)

<img src=img/weight_init_neron_demo.png width=240>


令$s=\sum_1^n w_i X_i$（$X_i$为输入数据的列向量），仅看隐藏层的第一个神经元，有n个输入神经元，对应n个权重w
$$
\begin{align}
Var(s)&=Var(\sum_{i=1}^n w_i X_i) \tag1 \\
&=\sum_{i=1}^n Var(w_i X_i) \tag2 \\
&=\sum_{i=1}^n w_i^2 Var(X_i) \tag3 \\
&=\left(\sum_{i=1}^n w_i^2 \right) Var(X) \tag4 \\
&=nVar(w) Var(X) \tag5
\end{align}
$$

推导$(2)$如下：
>
$$
Var\left( \sum_{i=1}^N X_i \right)=\sum_{i,j=1}^N Cov(X_i,X_j)=\sum_{i=1}^N Var(X_i) + \sum_{i \ne j} Cov(X_i, X_j)
$$
如果$X_1, \cdots, X_N$不相关，则$Cov(X_i, X_j)=0, \forall i\ne j$，
$$
Var\left( \sum_{i=1}^N X_i \right)=\sum_{i=1}^N Var(X_i)
$$

推导$(3)$对每一个列向量$X_i$对应的权重$w_i$都是固定值，可以提取出来

推导$(4)$假设每一个列向量$X_i$都有相同的方差，记为$Var(X)$

推导$(5)$如下：
>
$$
\begin{align}
Var(w)&= E[(w-E[w])^2] \\
&=E[w^2-2wE[w]+E[w]^2] \\
&=E[w^2]-2E[w]E[w]+E[w]^2 \\
&=E[w^2]-E[w]^2 &(w均值为0，E[w]=0) \\
&=E[w^2] \\
&={1 \over n} \sum_{i=1}^n w_i^2 \\
\end{align}
$$

##Artificially expanding the training data
如果训练数据太少，可以用如下方法扩展训练数据：

- rotate it by a small amount：例如对MNIST数据集旋转15°
- elastic distortions: a special type of image distortion intended to emulate the random oscillations found in hand muscles

##参考：

- [other_techniques_for_regularization](http://neuralnetworksanddeeplearning.com/chap3.html#other_techniques_for_regularization)
- [http://cs231n.github.io/neural-networks-2/](http://cs231n.github.io/neural-networks-2/)
- [知乎intelligentunit翻译](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

