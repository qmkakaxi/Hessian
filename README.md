# Hessia of model.parameters()

## 环境：
 1. python 3.5.2  
 2. pytorch 1.3.0 

## 本项目提供两种对模型参数对计算，海森矩阵(HessianMartix)和海森向量积(HessianVectorProduct)

1. 海森矩阵(HessianMartix)：
```
python HessianMarti.py
```

2. 海森向量积(HessianVectorProduct)：
```
python HessianVectorProduct.py
```

3. 实现方式：
 求导机制使用pytorch的torch.autograd,相关教程请看pytorch官方文档：https://pytorch.org/docs/stable/autograd.html
 
 

提醒：模型的参数数量一般都十分巨大，如非必要，请海森向量积(HessianVectorProduct)避免使用海森矩阵(HessianMartix)。

备注：如果您想对您对模型计算海森矩阵(HessianMartix)和海森向量积(HessianVectorProduct)，只需要加入在Nets.py中加入您的网络结构并且加载您对数据即可。
