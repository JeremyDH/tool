#PyTorch 分布式训练 nn.DistributedDataParallel
![图1](Mark_image/b1.png)
 1、分布式训练的几个选择： 
    (1).模型并行 VS 数据并行
    (2).PS 模型 vs Ring-Allreduce

 2、 假设 有两台机子，每台8张显卡，那就是2x8=16个进程，并行数是16。DDP 推荐每个进程一张卡，在16张显卡，16的并行数下，DDP会同时启动16个进程。rank表示当前进程的序号（0~16），用于进程间通讯。local_rank 表示每台机子上的进程的序号（0~7），也用来指定机器上的gpu序号。用起来就是一行代码，model = DDP(model, device_ids=[local_rank], output_device=local_rank)，后续的模型关于前向传播、后向传播的用法，和单机单卡完全一致。

 3、 每个进程跑的是同一份代码，进程从外界（比如环境变量）获取 rank/master_addr/master_port 等参数，rank = 0 的进程为 master 进程

 4、相关详细信息:

   (1) 加载模型阶段。每个GPU都拥有模型的一个副本，所以不需要拷贝模型。rank为0的进程会将网络初始化参数broadcast到其它每个进程中，确保每个进程中的模型都拥有一样的初始化值。

   (2)加载数据阶段。DDP 不需要广播数据，而是使用多进程并行加载数据。在 host 之上，每个worker进程都会把自己负责的数据从硬盘加载到 page-locked memory。DistributedSampler 保证每个进程加载到的数据是彼此不重叠的。

   (3)前向传播阶段。在每个GPU之上运行前向传播，计算输出。每个GPU都执行同样的训练，所以不需要有主 GPU。

   (4)计算损失。在每个GPU之上计算损失。

   (5)反向传播阶段。运行后向传播来计算梯度，在计算梯度同时也对梯度执行all-reduce操作。每一层的梯度不依赖于前一层，所以梯度的All-Reduce和后向过程同时计算，以进一步缓解网络瓶颈。在后向传播的最后，每个节点都得到了平均梯度，这样模型参数保持同步。关于AllReduce

   (6)更新模型参数阶段。因为每个GPU都从完全相同的模型开始训练，并且梯度被all-reduced，因此每个GPU在反向传播结束时最终得到平均梯度的相同副本，所有GPU上的权重更新都相同，也就不需要模型同步了

4、简明教程

	    import torch.distributed as dist
    	import torch.utils.data.distributed
    
    	parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
    	parser.add_argument('--rank', default=0,help='rank of current process')
    	parser.add_argument('--word_size', default=2,help="word size")
    	parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',help="init-method")
    	args = parser.parse_args()
    	...
    	dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)
    	...
    	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    	## 数据并行需要进行数据切片
    	train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=args.world_size,rank=rank)
    	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    	...
    	net = Net() 
    	net = Net().to(device) # device 代表到某个 gpu 或cpu 上
    	## 使用DistributedDataParallel 修饰模型
    	net = torch.nn.parallel.DistributedDataParallel(net)

  （1）用dist.init-process-group初始化分布式环境

      1. 一般来说，nccl 用于 GPU 分布式训练，gloo 用于 CPU 进行分布式训练。
      2. 这个调用是阻塞的，必须等待所有进程来同步，如果任何一个进程出错，就会失败。
  （2）数据侧，我们nn.utils.data.DistributedSampler来给各个进程切分数据，只需要在dataloader中使用这个sampler就好

      1. 使用 DDP 时，不再是从主 GPU 分发数据到其他 GPU 上，而是各 GPU 从自己的硬盘上读取属于自己的那份数据。
      2. 训练循环过程的每个epoch开始时调用train_sampler.set_epoch(epoch)，（主要是为了保证每个epoch的划分是不同的）其它的训练代码都保持不变。
      3. 举个例子：假设10w 数据，10个实例。单机训练任务一般采用batch 梯度下降，比如batch=100。分布式后，10个实例每个训练1w，这个时候每个实例 batch 是100 还是 10 呢？是100，batch 的大小取决于 每个实例 GPU 显存能放下多少数据。
   (3)模型侧，只需要用DistributedDataParallel包装一下原来的model

   (4)pytorch中的任何net都是torch.nn.Module的子类,DistributedDataParallel也是 torch.nn.Module 子类，任何torch.nn.Module 子类 都可以覆盖` __init__ `和 forward方法 ，DistributedDataParallel 可以从 net 拿到模型数据（以及在哪个gpu 卡上运行） ，也可以 从指定或默认的 process_group 获取信息。最后在`__init__ `和 forward 中加入 同步梯度的逻辑，完美的将 同步梯度的逻辑 隐藏了起来。
  
   实例代码：

	import os
	import torch
	import torch.nn as nn
	import torch.distributed as dist
	import torch.utils.data.distributed
	import torchvision
	import torchvision.transforms as T

	BATCH_SIZE = 256
	EPOCHS = 5
	transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5))]
	)
	if __name__ == '__main__':
    #0. set up distribute device environment
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) #把模型放置到特定的GPU上
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    #1、define network
    net = torchvision.models.resnet18(pretrained=False, num_class=10)
    net = net.to(device)
    #DistributedDataParallel
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # DistributedSampler
    # we test single Machine with 2 GPUs so the [batch size] for each process is 256 / 2 = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size = BATCH_SIZE,
        num_workers = 4,
        pin_memory = True,
        sampler = train_sampler,
    )
    #3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr = 0.01 * 2,
        momentum = 0.9,
        weight_decay = 0.0001,
        nesterov = True,
    )
    if rank == 0:
        print("            =======  Training  ======= \n")
    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS+1):
        train_loss = correct = total = 0
        #set sampler
        train_loader.sampler.set_epoch(ep)
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad() #将梯度归零
            loss.backward() #反向传播计算得到每个参数的梯度值
            optimizer.step()  # 通过梯度下降执行一步参数更新，optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。
            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
            #输出loss 和正确率
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")