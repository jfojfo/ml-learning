##参考
* [参考1](http://www.cnblogs.com/lancelod/p/3863053.html)
* [参考2](https://community.qingcloud.com/topic/464/deep-learning-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%E6%95%B4%E7%90%86%E7%B3%BB%E5%88%97%E4%B9%8B-%E5%85%AD)
* [参考3(知乎)](https://zhuanlan.zhihu.com/p/20431433)
* [github原始出处](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/7_Restricted_Boltzmann_Machine_%E5%8F%97%E9%99%90%E6%B3%A2%E5%B0%94%E5%85%B9%E6%9B%BC%E6%9C%BA.md)
* [参考4](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNEquations)

##公式
$$
E(v,h)=-b'v-c'h-v'Wh \\
E(v,h)=- \sum_i b_i v_i - \sum_j c_j h_j -\sum_{ij} v_i W_{ij} h_j \\
\theta=\{W,b,c\} \\
\begin{aligned}
P_{\theta}(v,h)={{exp}^{-E(v,h)} \over Z(\theta)} &= {1\over Z(\theta)} \prod_i e^{b_i v_i} \prod_j e^{c_j h_j} \prod_{ij} e^{v_i W_{ij} h_j} \\
&= {1\over Z(\theta)} \prod_i e^{b_i v_i} \prod_j e^{(c_j+\sum_i v_iW_{ij}) h_j} \\
\end{aligned} \\
\begin{aligned}
P_{\theta}(v)=\sum_h P_{\theta}(v,h) &= {1\over Z(\theta)} \prod_i e^{b_i v_i} \prod_j \sum_{h_j} e^{(c_j+\sum_i v_iW_{ij}) h_j} \\
&= {1\over Z(\theta)} e^{b'v} \prod_j \sum_{h_j} e^{(c_j+ v'W_{*j}) h_j} \\
\end{aligned} \\
P_{\theta}(v)={1\over Z(\theta)} e^{-F(v)} \\
F(v)=-b'v-\sum_j log \sum_{h_j} e^{(c_j+ v'W_{*j}) h_j} \\
-{\partial log P_\theta(v) \over \partial \theta}= {\partial F(v) \over \partial \theta}-\sum_{\hat x} p_\theta(\hat x) {\partial F(\hat x) \over \partial \theta}\\
L(\theta)={1\over N}\sum_{n=1}^N log P_{\theta}(v^{(n)}) \\
{\partial L(\theta) \over \partial W_{ij}}=
$$