#Pytorch 分布式训练迁移
##Pytorch数据并行接口
Pytorch提供了两种并行训练接口：torch.nn.DataParallel简称(DP)和torch.nn.DistributedDataParallel简称(DDP)。

* 1、torch.nn.DataParallel
  > model = torch.nn.DataParallel(model)
  > 优点：使用方法简单，只需在单卡训练的基础上，使用DP接口封装一次model即可。
  
  > 缺点：
  > 1、只能单机使用。
  > 2、0号加速卡压力加大，负载不均匀，整体并行度不高。
  > 3、单进程，受到python GIL锁强限制，效率不高。
 
* 2、torch.nn.DistributedDataParallel
  > DDP使用多进程方式，每个加速卡由一个进程控制，每个加速卡一份完整的模型，主机CPU将不同的数据分发到每个加速卡上，加速卡之间只传递反向求导的梯度信息。

  > 优点：可以多机部署。②每个加速卡无主次之分，负载均匀，并行度高。③每个加速卡对应一个python解析器，没有python GIL的限制，效率较高
 
  > 缺点：使用相对复杂。

#Pytorch分布式训练NPU迁移（DDP）
Pytorch提供两种DDP启动方式，torch.multiprocessing.spawn接口函数和 torch.distributed.launch启动器。由于启动器在部分进程异常退出时，其他进程需要手动关闭的毛病，NPU迁移一般使用torch.multiprocessing.spawn接口函数。

###代码示例
	import argparse
	import torch
	import torchvision
	import os
	import torch.distributed as dist
	from torch.nn.parallel import DistributedDataParallel as DDP
	class LinearModel(torch.nn.Module):
	    def __int__(self):
	        super(LinearModel, self).__init__()
	        self.linear = torch.nn.Linear(1, 1)
	    def forward(self, x):
	        y_pred = self.linear(x)
	        return y_pred
	
	def train(npu, ngpus_per_node, opt):
	    device = torch.device('npu:%d'%npu)
	    opt.world_size = ngpus_per_node * opt.world_size #总的加速卡数量
	    opt.local_rank = opt.local_rank * ngpus_per_node + npu # 当前卡在所以训练卡中的编号
	    dist.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.local_rank) # 设置后端，聚合通信算法
	    train_dataset = torchvision.datasets.MNIST('./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)  # 读取训练集
	    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # 划分数据集，以避免不同进程间数据重复
	    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)  # 训练集迭代器
	    test_dataset = torchvision.datasets.MNIST('./mnist', train=False, transform=torchvision.transforms.ToTensor())  # 读取验证集
	    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)  # 验证集迭代器
	    model = LinearModel().to(device) #模型初始化
	    criterion = torch.nn.MSELoss(size_average=False) #均方误差损失函数
	    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #梯度下降优化
	    model = DDP(model, device_ids=[npu], broadcast_buffers=False)
	
	    for epoch in range(opt.epochs):
	        train_sampler.set_epoch(epoch) #打乱数据集
	        model.train() #训练模式
	
	        for imgs, labels in train_loader:
	            imgs, labels = imgs.to(device),labels.to(device)
	            optimizer.zero_grad()
	            pred = model(imgs)
	            loss = criterion(pred, labels)##  ##
	            loss.backward()        #反向梯度计算
	            optimizer.step()       #模型参数更新
	    with torch.no_grad():
	        for imgs in test_loader:
	            preds = model(imgs)
	if __name__ == '__main__':
	    parser = argparse.ArgumentParser()
	    parser.add_argument('--epochs', type=int, default=300)
	    parser.add_argument('--device', type=int, default=0)
	    # DDP parameters
	    parser.add_argument('--world-size', default=1, type=int, help='number of nodes')          # 训练服务器数量
	    parser.add_argument('--device-num', default=1, type=int, help='number of cards per node') # 每台服务器的卡数量
	    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')           # 训练服务器编号
	    parser.add_argument('--addr', default='10.38.60.25', type=str, help='master IP')          # 训练主节点IP
	    opt = parser.parse_args()
	    os.environ['MASTER_ADDR'] = opt.addr                                                # 设置环境变量：训练主节点IP
	    os.environ['MASTER_PORT'] = '29501'                                                 # 设置环境变量：训练主节点端口
	    ngpus_per_node = opt.device_num                                                     # 每台服务器的加速卡数量
	    torch.multiprocessing.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))