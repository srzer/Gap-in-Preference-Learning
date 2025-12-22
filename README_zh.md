# Understanding the Performance Gap in Preference Learning
本仓库提供了论文["Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO"](https://arxiv.org/pdf/2505.19770)的代码实现。主要分为两个部分 1) *对论文第三节结论的验证*，实现了在线DPO和简洁RL的对比；2) *对论文第四节结论的验证*，实现了DPO和奖励建模的对比。 ***如果你在复现中发现任何问题，请提出issue，我们会及时回复！***

[English](./README.md)

## Exp-0: 对论文第三节结论的验证

```bash
cd Exp-0
```

### 🔨 基础设置
这一部分主要基于[Online-RLHF](https://github.com/RLHFlow/Online-RLHF)。

```bash
create -n rlhflow python=3.10
conda activate rlhflow
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 🏄 工作流

#### 🐜 模型初始化

你可以直接使用我们在huggingface上公开的初始模型`zhezi12138/gpt2-large_sft_model`。下载并保存为`models/sft_model`即可。我们也提供了SFT的训练脚本：`scripts/safe_rlhf/sft.sh`.

#### 🐝 迭代数据生成与标注

DPO模型的生成：
```bash
bash scripts/safe_rlhf/gen_dpo.sh ${iter_number} ${#responses per prompt} # the response number is set as 2 in our experiments
```

RL模型的生成：
```bash
bash scripts/safe_rlhf/gen_pg.sh ${iter_number} ${#responses per prompt} # the response number is set as 2 in our experiments; when iter_number=1, it is equivalent to DPO generation.
```

DPO模型生成数据的标注：
```bash
bash scripts/annotate.sh ${iter_number} dpo ${responses per prompt} # the response number is set as 2 in our experiments
```

我们还在huggingface上公开了一个弱奖励模型 `zhezi12138/weak_rm_gpt2-large_harmless`，来模拟奖励模型误设的情况。

RL模型生成数据的标注：
```bash
bash scripts/annotate.sh ${iter_number} pg ${responses per prompt} # the response number is set as 2 in our experiments
bash scripts/annotate_weak.sh ${iter_number} pg ${responses per prompt} # annotation with reward model mis-specification
```

#### 🐧 迭代训练

在线DPO的训练：
```bash
conda activate rlhflow
bash scripts/safe_rlhf/dpo_online.sh ${iter_number} # no policy model mis-specification
bash scripts/safe_rlhf/dpo_online_mis.sh ${iter_number} # with policy model mis-specification
```

RL的训练：
```bash
conda activate rlhflow
bash scripts/safe_rlhf/pg.sh ${iter_number} # no policy model mis-specification
bash scripts/safe_rlhf/pg_mis.sh ${iter_number} # with policy model mis-specification
```

#### 🐤 评估
测试DPO/RL模型的指令为：
```bash
bash scripts/safe_rlhf/gen_test.sh 0 ${here `dpo` or `pg`} ${iter_number}
bash scripts/annotate_test.sh 0 ${here `dpo` or `pg`} ${iter_number}
```

最后，我们也提供了迭代训练工作流的多合一脚本：`scirpts/pipeline_train.sh` 和 `scripts/pipeline_train_mis.sh`，以及评估脚本：`scripts/pipeline_test.sh`。

## Exp-1: 对论文第四节结论的验证

```bash
cd Exp-1
```

### 🔨 基础设置
这一部分主要基于[modpo](https://github.com/ZHZisZZ/modpo)。

```bash
create -n rml python=3.10
conda activate rml
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 🏄 工作流

#### 🍎 DPO训练

```bash
bash scripts/dpo/run.sh ${data size} ${seed number}
```
在我们的论文中，我们主要采用了1000,2000,4000,9000的数据规模，和41,42,43的随机种子。

#### 🍏 奖励建模训练
 
```bash
bash scripts/rm/run.sh ${data size} ${seed number}
```
在我们的论文中，我们主要采用了1000,2000,4000,9000的数据规模，和41,42,43的随机种子。

*请注意训练采用的默认偏好目标为'better'，你可以修改脚本的第三行来将其改成'safer'。*

在训练结束时，评估准确率会被自动报告。

## 🏷️ 证书
本仓库使用MIT证书。

## 📝 Citation
如果我们的工作对你的研究有帮助，可以考虑引用本论文，谢谢：

```
@article{shi2025understandingperformancegappreference,
      title={Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO}, 
      author={Ruizhe Shi and Minhak Song and Runlong Zhou and Zihan Zhang and Maryam Fazel and Simon S. Du},
      year={2025},
      journal={arXiv},
      url={https://arxiv.org/abs/2505.19770}, 
}
```
