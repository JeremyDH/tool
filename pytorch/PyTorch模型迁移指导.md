#将原生pytorch的模型代码适配到Ascend-Pytorch
##step1 迁移前的准备
 关于分布式：由于NPU 的限制，pytorch需要DDP,若原始代码使用的是DP需要修改为DDP,DP的一些相应实现例如torch.cuda.common，则可以替换为torch.distributed相关操作，其中静态loss scale是核心
 ![图1](Mark_image/p1.png)
##step2 单P模型迁移
 ![图2](Mark_image/p2.png)
##step3 多P模型迁移
 ![图3](Mark_image/p3.png)
##Extra 报错排查
 ![图4](Mark_image/p4.png)


