# REFLOW

## data

创建软连接 data/coco2014_reflow 链接到存放数据的文件夹

使用 reflow_generate_data.py 产生数据。(如果要使用 oneflow 框架，请激活安装了 oneflow 和 oneflow-diffusers 的虚拟环境，并设置 use_oneflow=True . 环境创建教程见 [安装oneflow-diffusers](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion#without-docker) )

模型默认使用 AltDiffusion , DPMSolverMultistepScheduler ; 数据默认为 coco2014 数据集

数据产生过程为：
1. 随机抽取数据集中的 caption . 
2. 产生随机噪声 z0 . 
3. 将 caption 和 z0 输入扩散模型，推理得到 z1 . 
4. 保存 caption, z0, z1 . (所有 caption 统一保存到 .txt 文件，每一对 (z0,z1) 保存为 npy 文件)

在函数 prepare_args 中指定所有的参数设置。参数列表为：
- infer_steps . 扩散模型推理步数。
- seed . 随机种子。
- save_dir . 存放所有产生内容的根目录。
- split . ["train","val"] . 使用 coco 的训练集或验证集的 caption. 
- devices . list 类型。指定要使用的 gpu 编号。
- total_nums . 需要产生的数据数量。
- bs . 批量大小。
- part . 数据量过大时，分批产生数据使用。

### 创建 lmdb 文件格式

可以使用 reflow/data/utils.py 中的 data2lmdb 函数将 npy 文件打包成 lmdb 文件格式。需要指定参数：
- dpath . npy 文件存放的根目录。例如：`data/coco2014_reflow/test/content/images` . 

执行后，会在与 dpath 平行的目录下创建 lmdb 目录，存放 .lmdb 数据库文件。

## train

使用 reflow_train_ddp.py 脚本执行. 请使用 accelerate 模块启动该脚本。accelerate 用法见：[使用accelerate执行分布式训练脚本](https://huggingface.co/docs/accelerate/v0.16.0/en/basic_tutorials/launch#using-accelerate-launch)

启动脚本时指定如下参数：
- config . py文件，存放实验的所有主要参数
- workdir . log 目录
- comment(optional) . 实验注释，标注一些特别的更改。

默认 train 脚本的主要参数存放于 reflow/configs/train.py 。如下参数需要特别注意：
- diffusers.load_score_model . bool 类型。是否加载 diffusers unet 模型的权重来初始化 score model . 
- training.randz0 . ['random','fix'] . 指定为 fix 则使用数据集中的 z0 数据；random 则 z0 重新随机采样
- training.ckpt_path . 从accelerate 的检查点恢复训练。文件夹，命名为 `checkpoint_s{step}`
- sampling.randz0 . ['random','fix'] . 含义同 training.randz0
- sampling.use_ode_sampler . ['euler','rk45'] . 选择采样模式。
- sampling.sample_N . 采样过程执行步数。(仅对 'euler' 采样器有效。)
- reflow.reflow_t_schedule . t0, t1, uniform, or an integer k > 1 . 
- reflow.reflow_loss . l2, lpips, lpips+l2

## sample

使用 reflow_sample.py 脚本执行.

启动脚本时指定如下参数：
- config . py文件，存放实验的所有主要参数
- eval_folder . sample 目录

默认 sample 脚本的主要参数存放于 reflow/configs/sample.py 。如下参数需要特别注意：
- sampling.decode_noise . bool . 是否解码产生 noise 的图片。
- sampling.decode_latent . bool . 是否解码产生 latent 的图片。
- sampling.return_traj . bool . 是否打印采样过程的 trajectory . 推理步数较大时请关闭改参数。
- sampling.randz0 . 含义同 training.randz0
- sampling.ckpt_path . 加载保存的 score model . .pth 文件, 命名为 `score_model_s{step}.pth`

