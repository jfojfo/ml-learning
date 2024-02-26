#Spase coding
## 说明
$s_{m\times n}A_{n\times k}=y_{m\times k}$

参考：http://www.cnblogs.com/tornadomeet/archive/2013/04/16/3024292.html

##代价函数
非拓扑结构：
$$
\begin{align}
J(A,s) = {1\over 2m}\sum \left(sA+b-y\right)^2 +{1\over 2}\gamma\sum A^2 + {1\over 2}\lambda\sum\sqrt{s^2+\varepsilon} \\
\end{align} \tag1
$$

拓扑结构：
$$
\begin{align}
J(A,s) = {1\over 2m}\sum \left(sA+b-y\right)^2 +{1\over 2}\gamma\sum A^2 + {1\over 2}\lambda\sum\sqrt{s^2V+\varepsilon} \\
\end{align} \tag2
$$
$V$是Group Matrix，将相邻的几个$s^2$求和连在一起

##Gradient
$$
{\partial J(A,s) \over \partial A}={1\over m} s^T \left(sA+b-y\right) + \gamma A
\tag3 $$

$$
{\partial J(A,s) \over \partial b}={1\over m} \sum_i \left(sA+b-y\right)_{ij}
\tag4 $$

非拓扑结构：
$$
{\partial J(A,s) \over \partial s}={1\over m}\left(sA+b-y\right) A^T + {1\over 2}\lambda {s \over \sqrt{s^2+\varepsilon}}
\tag5 $$

拓扑结构：
$$
{\partial J(A,s) \over \partial s}={1\over m}\left(sA+b-y\right) A^T + {1\over 2}\lambda s \bullet {1 \over \sqrt{s^2V+\varepsilon}} V^T
\tag6 $$

注意：$(6)$中的$\bullet$表示前后两矩阵的每个元素相乘（element-wise multiply），而不是矩阵相乘！

##快速求A
$$
A = (s^Ts+m\gamma I)^{-1} s^T(y-b)
\tag7 $$
$$
b = {1\over m} \sum_i \left(y-sA\right)_{ij}
\tag8 $$

$令{\partial J(A,s) \over \partial A}=0$，得
$$
\begin{align}
0 &= {1\over m}s^T(sA+b-y)+\gamma A \\
0 &= s^TsA+s^T(b-y) + m \gamma A \\
(s^Ts+ m \gamma I)A &= s^T(y-b) \\
A &= (s^Ts+m\gamma I)^{-1} s^T(y-b) \\
\end{align}
$$

$令{\partial J(A,s) \over \partial b}=0$，得
$$
{1\over m} \sum_i \left(sA+b-y\right)_{ij} = 0 \\
b = {1\over m} \sum_i \left(y-sA\right)_{ij}
$$



