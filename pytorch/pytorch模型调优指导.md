#pytorch模型调优指导
##性能工具：pytorch profiling, CANN Profiling
![图1](Mark_image/p5.png)
![图1](Mark_image/p6.png)
##性能优化方法: FUSE+COMBINE、taskset、BS、优化库、Autotune
![图1](Mark_image/p7.png)
##精度优化方法：Loss_scale、LR、model_eval()
![图1](Mark_image/p8.png)
##有关Loss函数变化曲线和模型超参数的关系
	**train loss 不断下降，test loss不断下降，说明网络仍在学习;
	train loss 不断下降，test loss趋于不变，说明网络过拟合;
	train loss 趋于不变，test loss不断下降，说明数据集100%有问题;
	train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;
	train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。**
