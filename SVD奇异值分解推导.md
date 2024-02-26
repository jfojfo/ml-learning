#SVD奇异值分解推导

##分解形式
$$ A=U \Sigma V^T $$

##协方差矩阵
$A_{m\times n}$，$m$倍协方差矩阵$S$，$V$特征向量，$\lambda$特征值矩阵
$$
\begin{aligned}
& S=A^TA=V\lambda V^T \\
& Sv_i=\lambda_iv_i=v_i\lambda_i\\
& SV=V\lambda\\
& V^T=V^{-1}
\end{aligned}
$$

令$A \rightarrow AV$，
$$
S \rightarrow V^TA^TAV=V^TA^TAV=V^TSV=\lambda
$$
可以看到，$A$变换成$AV$后，新协方差矩阵变为$\lambda$

##左奇异矩阵$U$
$ 令 u_i={Av_i\over \sigma_i}，\sigma_i=\sqrt\lambda_i $，则
$$ u_i^Tu_j={v^T_iA^T\over \sigma_i}{Av_j\over \sigma_j}={v_i^TSv_j\over\sigma_i\sigma_j}＝{v_i^Tv_j\lambda_j\over\sigma_i\sigma_j} $$

$$
\begin{cases}
u_i^Tu_i={v_i^Tv_i\lambda_i\over\lambda_i}=1(1\le i\le n) \\
u_i^Tu_j=0(1\le i\le n,i\ne j)
\end{cases} \tag1
$$
$$
U\Sigma=AV，其中\Sigma=
\begin{bmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_m \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
$$
$$ A=U\Sigma V^T $$

由$(1)得$$U$是标准正交基组成的矩阵，下面进一步推导出$AA^T=U\lambda U^T$，说明$U$是$AA^T的特征向量$，特征值也是$\lambda$

$$
U^TS^TU=U^TAA^TU=U^TU\Sigma V^TV\Sigma^TU^TU=I\Sigma I\Sigma^TI=\lambda \\
S^T=AA^T=U\lambda U^T
$$

##右奇异矩阵$V$
与左奇异矩阵$U$类似，令$ v_i = {A^Tu_i \over \sigma_i}，\sigma_i=\sqrt\lambda_i $，同样可得：
$$ V\Sigma^T=A^TU $$
$$ A^T=V\Sigma^TU^T $$
$$ A=(V\Sigma^TU^T)^T=U\Sigma V^T $$

##$U$、$V$变换含义
$A_{m\times n}(m>n)$，理解为$m$行样本数据，$n$列特征值

####$AV=U\Sigma$将数据投影到新的混合特征值空间
- $AV$相当于将$A$的每一行样本数据（行向量）变换到$V：\{v_1,v_2,\cdots,v_n\}$坐标
- $v_1$对应特征值$\lambda_1$最大，包含能量最大，变换到该方向的数据最重要
- 原特征值空间（列空间）变换后，按照从主到次，在$V：\{v_1,v_2,\cdots,v_n\}$上重新分布
- **列压缩**，将变换到$v_n、v_{n-1} \cdots$末尾次要方向的数据去除，实现特征值空间（列空间）降维、数据压缩、降噪
- 变换后，形成新的**混合特征值空间**，第一列$\sigma_1 u_1$是**主混合特征值**
- ${(u_1)}_1\sigma_1=<A_{1,1:n},v_1>=U_{1,1}\sigma_1$（向量内积后$\Sigma$累加）,混合了第一行样本数据在各个特征维度的特征值，依次混合第二行、第三行...，形成一列新的**混合特征值**$\sigma_1u_1$
- **主混合特征值**方向上的新样本数据组合形成新向量$\sigma_1u_1$

####$U^TA=\Sigma V^T$将数据投影到新的混合样本数据空间
- $U^TA$相当于将$A$的每一列特征值（列向量）变换到$U：\{u_1,u_2,\cdots,u_n\}$坐标
- 原样本数据空间（行空间）变换后，按照从主到次，在$U：\{u_1,u_2,\cdots,u_n\}$上重新分布
- **行压缩**，将变换到$u_n、u_{n-1} \cdots$末尾次要方向的数据去除，实现样本数据空间（行空间）降维、数据压缩、降噪
- 变换后，形成新的**混合样本数据空间**，第一行$\sigma_1 v_1^T$是**主混合样本数据**
- ${(v^T_1)}_1\sigma_1=<A_{1:m,1},u_1>=U_{1,1}\sigma_1$（向量内积后$\Sigma$累加），混合了第一列特征值的每个样本数据，依次混合第二列、第三列...，形成一行新的**混合样本数据**$\sigma_1v_1$
- **主混合样本数据**方向上的新特征值组合形成新向量$\sigma_1v_1$

