#Pytorch模型迁移至NPU
##一、项目简介
###1、简单信息
* 模型名：3DMPPE_ROOT NET
* 算法场景：3D 姿态估计
* 论文链接：[https://arxiv.org/abs/1907.11346](https://arxiv.org/abs/1907.11346)
* 代码链接：[https://github.com/mks0601/`3DMPPE_ROOTNET_RELEASE`](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)
* 目标配置：RootNet
* 参考精度 ：MuCo
* 目标数据集：AP_25（percentage）31.0

###2.训练阶段里程碑
* Step 0: 本地模型调试
* Step 1: 1 卡 GPU 精度和性能复现
* Step 2: 8 卡 GPU 精度和性能复现
* Step 3: 1 卡 NPU 精度和性能对齐
* Step 4: 8 卡 NPU 精度和性能对齐
##二、模型训练1P和8P按照github仓库中的的方法，达到精度
在单卡训练例:

    python train.py --gpu 0
在多卡训练：

    python train.py --gpu 0-7
##三、模型迁移-本地代码添加
###1、添加prof、FPS、apex
####(1)prof:
* Pytorch中autograd 的 profiler 提供内视每个操作在GPU和CPU的花销，通过对生成的.prof文件进行分析可以发现耗时较为高的操作或存在问题的算子，从而进行替换或修改以提高性能。
* 注意prof仅用于调试、相当于Debug模式，想要得到prof文件，跑一个epoch即可。正常训练时不需要使用prof，在后续性能调优部分会有对prof文件的分析，使用prof会使代码运行时间大量提升。同时prof仅仅用于单卡，即一张GPU或NPU，如果在多卡环境下启用prof会导致RAM一直上升直到爆内存。
* 相关代码：

>     with torch.autograd.profiler.profile(use_cuda=True) as prof: 
> 	    out = model(input_tensor)
> 	    loss = loss_func(out)
> 	    optimizer.zero_grad()
> 	    if amp_mode:
> 	        with amp.scale_loss(loss, optimizer) as scaled_loss:
> 	            scaled_loss.backward()
> 	    else:
> 	        loss.backward()
> 	    optimizer.step()
> 
> 	print(prof.key_averages().table(sort_by="self_cpu_time_total"))
> 	prof.export_chrome_trace("output.prof") # "output.prof"为输出文件地址

* 为了方便后续在控制台运行代码，符合代码规范，可以添加if语句进行判断。
> 	    if cfg.use_prof:
>          with torch.autograd.profiler.profile(use_npu=True) as prof:
>     	       out = model(input_tensor)
>     	   ...
>     	    # Train model
>     	else：
>     
>     	    out = model(input_tensor)
>     	    ...
>     	   # Train model

* 打开方式：然后打开chrome或edge浏览器，输入**  chrome://tracing/**  ，然后加载生成的json文件
####(2)FPS:
* 公式： `FPS = BatchSize * num_devices / time_avg`
* 请注意此处的`time_avg`不要统计每个`epoch`的前5个`step`的耗时，因为NPU前几个step会进行数据读取，较为耗时。`time_avg`可以使用以`epoch`为单位的平均耗时，需要去除前几个step加载数据带来的影响，统计类示例请参考AverageMeter
####(3)Apex

>  APEX是英伟达开源的，完美支持PyTorch框架，用于改变数据格式来减小模型显存占用的工具。其中最有价值的是amp（Automatic Mixed Precision），将模型的大部分操作都用Float16数据类型测试，一些特别操作仍然使用Float32。并且用户仅仅通过三行代码即可完美将自己的训练代码迁移到该模型。实验证明，使用Float16作为大部分操作的数据类型，并没有降低参数，在一些实验中，反而由于可以增大Batchsize，带来精度上的提升，以及训练速度上的提升。

示例代码：

	# Declare model and optimizer as usual, with default (FP32) precision
	model = torch.nn.Linear(D_in, D_out).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
	
	# Allow Amp to perform casts as required by the opt_level
	# 注意，amp.initialize必须在DDP(DistributedDataParallel)之前，否则多卡时会有报错提示
	model, optimizer = amp.initialize(model, optimizer, opt_level="O2"，loss_scale=128.0)
	...
	# loss.backward() becomes:
	with amp.scale_loss(loss, optimizer) as scaled_loss:
	    scaled_loss.backward()
	..
注意事项：

>   推荐优先使用 `opt_level=‘O2’`, `loss_scale=128.0 `的配置进行`amp.initialize `若无法收敛推荐使用
>   
>	`opt_level=‘O1’`, `loss_scale=128.0 `的配置进行amp.initialize 若依然无法收敛推荐使用
>	
>   `opt_level=‘O1’`, `loss_scale=None` 的配置进行amp.initialize   
>   
>   实际运行过程中，O2相较于O1可以带来性能上更多的提升，但也会对应地损失一些精度，如果不确定`loss_scale`也可以，也可以将其设置为-1，表示动态，系统会自动调整。关于`opt_level`和`loss_scale`的调整会在后续性能和精度优化部分进行阐述。添加完上述三个主要部分后，需要再次在本地进行代码调试和运行，复现论文的精度，正常情况下在添加了apex后，性能会有所提升，精度会略有下降。
####(4)bottleneck功能
pytorch已经提供了更为方便的工具，位于torch.utils.bottleneck， 命令行：

    python3 -m torch.untils.bottleneck trian.py
不需要修改train.py的代码，直接执行上述指令就可profiling出数据来！

#### 添加DDP
此部分参见   ` pytorch_分布式训练_nn.DistributedDataParallenl`

####多卡环境下的Batch_size、FPS、以及模型保存相较于单卡还需要额外进行一些改变
关于`batch_size`,例如在1p环境中的值为`512`， 在8P中则为 `512*8`，依此类推在16P中则为 `512*16`。
关于FPS的计算：
FPS原先为：`cfg.batch_size / fps_timer.average_time` 多卡环境下为：`cfg.batch_size * ngpus_per_node / fps_timer.average_time`。
模型保存时只需保存一个，所以需要添加`if dist.get_rank() == 0:`
`batch_size`方面，多卡情况下输入的是总的`batch_size`，每张卡的`batch_size`需要进行额外计算：`args.batch_size = int(args.batch_size / ngpus_per_node)`。以上均为例子，修改需要根据代码实际情况，请多参考已经完成的模型或文档。
