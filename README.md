# LogisticsRegression
##20行python搞定逻辑回归

逻辑回归是在工业界广泛应用的分类算法，特点是结构简单，也因此有以下优缺点：

***pros***
1.训练和运行速度都很快
2.实现方便
3.内存占用少
4.可解释性好

***cons***
1.由于于无法拟合非线性关系，对特征工程的要求较高。
2.对多重共线性数据较为敏感


**结构上，逻辑回归和线性回归的区别仅仅在于增加了sigmoid函数（又称逻辑函数），将输出值转化为归一化的数值。**

线性回归
$$
Y=X \cdot W+B
$$
逻辑回归
$$
Y=\sigma (X \cdot W+B)
$$
其中
$$
\sigma(t)=\frac{1}{1+e^{-t}}
$$
为sigmoid函数

$$
\hat y=\left\{\begin{array}{ll}
0, & z<0 \\
0.5, & z=0 \\
1, & z>0
\end{array}, \quad z=w^{T} x+b\right.
$$

之所以加入sigmoid函数，是因为线性回归的输出值不在0~1之间，也不能拟合离散变量。
而理想的分类函数不可微。对数几率函数则是任意阶可导的凸函数，具有良好数学性质。另外，引入sigmoid函数可以连续地表示可能性的大小，不过输出值并不是数学意义上的“概率”。

**模型已经确定，接下来考虑参数，使用统计中极大似然估计的办法。**
伯努利分布下的最大似然估计推导出交叉熵损失函数：
假设
$$
\begin{array}{l}
P(Y=1 \mid x)=p(x) \\
P(Y=0 \mid x)=1-p(x)
\end{array}
$$
伯努利分布的概率密度函数
$$
f_{X}(x)=p^{x}(1-p)^{1-x}=\left\{\begin{array}{ll}
p & \text { if } x=1 \\
1-p & \text { if } x=0
\end{array}\right.
$$
似然函数
$$
L(w)=\prod\left[p\left(x_{i}\right)\right]^{y_{i}}\left[1-p\left(x_{i}\right)\right]^{1-y_{i}}
$$
为了便于计算改写为对数形式，发现与交叉熵完全一致
$$
\ln L(w)=\sum\left[y_{i} \ln p\left(x_{i}\right)+\left(1-y_{i}\right) \ln \left(1-p\left(x_{i}\right)\right)\right]
$$
取反后作为损失函数，最大化似然函数等价于最小化损失函数。
$$
J(w)=-\frac{1}{N} \ln L(w)
$$

本文中使用梯度下降法寻找最优参数
计算损失函数对于权重的偏导数，证明从略
$$
\frac{\partial J(w)}{\partial w_{i}}=\left(p\left(x_{i}\right)-y_{i}\right) x_{i}
$$
详细推导过程可以参考 https://blog.csdn.net/jasonzzj/article/details/52017438
更新参数
$$
w_{i}^{k+1}=w_{i}^{k}-\alpha \frac{\partial J(w)}{\partial w_{i}}
$$
