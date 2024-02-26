
神经网络/CNN/：

- 《Visualizing and Understanding Convolutional Networks》
- 《GradCAM - Visual Explanations from Deep Networks via Gradient-based Localization》
- 《CAM - Learning Deep Features for Discriminative Localization》
- 《Deep Inside Convolutional Networks- Visualising Image Classification Models and Saliency Maps》


《CAM...》的做法其实就是slim中resnet-v2的做法，最后一个conv层之后，加上一个average pool(resnet-v2的pool5)，然后通过1x1的conv得到logit，然后softmax。conv层输出为特征map $M$，1x1的conv的filter尺寸为1x1x2048x1000，squeeze之后得到$W_{ij}$，固定某个类别后，得到权重$w_i$，CAM的heat map为：

$$
HeatMap=\Sigma_{i=1}^{2048} M_i w_i
$$

《GradCAM...》的做法比CAM更通用，直接求最后一个conv层输出特征的gradient，得到$Grad_{i,j,n}$，高度为i，宽度为j，n个特征图，求和得权重w

$$
w_n = \Sigma_{i,j} Grad_{i,j,n}
$$

《Deep Inside...》这篇文章提到泰勒展开，对理解很有帮助。导数相当于泰勒展开后一次项的权重系数，权重越大，影响越大。

