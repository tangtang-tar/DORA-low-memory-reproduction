# 阶段 A：环境与资源准备

阶段 A 已完成。当前边界是：可以分别运行 Policy 和 BGE-M3，也已下载 PRM；不在 8 GB 显存上同时加载三者。

## 本地操作边界

- 项目、Conda 环境、模型、数据集、缓存、临时文件和结果都位于 `$DORA_ROOT`。
- 只调用系统已有的 `an existing Conda installation` 来启动 Conda，没有在那里新建环境或缓存。
- 没有安装系统级 CUDA，也没有修改 NVIDIA 驱动。
- `Qwen2.5-Math-PRM-7B` 约 15 GB，阶段 A 只校验四个权重分片，不在 8 GB 显存上加载。

## 每次开始工作

```bash
source $DORA_ROOT/activate_dora.sh
```

`source` 的作用是让环境变量留在当前终端。脚本会激活 DORA 内的 Conda 环境、把各种缓存指向 DORA，并进入源码目录。

## 独立测试

```bash
python $DORA_ROOT/scripts_local/test_policy.py
python $DORA_ROOT/scripts_local/test_embedding.py
```

两条命令要依次运行，不要并行。第一条加载 1.5B Policy 并生成一道数学题；进程退出释放显存后，第二条加载 BGE-M3 并输出语义相似度矩阵。

## 已固定的核心版本

- Python 3.10.21
- PyTorch 2.5.1 + CUDA 12.4 runtime
- vLLM 0.6.5
- Transformers 4.47.1
- Datasets 3.1.0
- FlagEmbedding 1.3.4

## 本地资源

- Policy：`models/Qwen2.5-1.5B-Instruct`
- PRM：`models/Qwen2.5-Math-PRM-7B`
- Embedding：`models/bge-m3`
- 数据集：`datasets/MATH-500`，共 500 行

## 官方源码的最小修复

官方仓库的 `scripts/modeling_qwen2_rm.py` 引用了仓库中不存在的 `configuration_qwen2_rm.py`。已将这一行改为从当前固定版本的 Transformers 导入等价的 `Qwen2Config`。修复后 `sal.search.dora` 和 `sal.models.reward_models` 均可正常导入。
