# Understanding the Performance Gap in Preference Learning
This repo provides code for the paper ["Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO"](https://arxiv.org/pdf/2505.19770). The codebase is composed of two parts: 1) *verifications of Section 3*, which implements online DPO and simple RL; 2) *verifications of Section 4*, which implements DPO and reward modeling. ***If you find any issue in reproduction, feel free to create an issue!***

## Exp-0: Verifications of Section 3

```bash
cd Exp-0
```

### 🔨 Set up
This part is directly based on [Online-RLHF](https://github.com/RLHFlow/Online-RLHF).

```bash
create -n rlhflow python=3.10
conda activate rlhflow
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 🏄 Pipeline

#### 🐜 Model initialization

You could directly use our released SFT model `zhezi12138/gpt2-large_sft_model` on huggingface as the initial model. Download it and save as `models/sft_model`. The SFT script is also provided as `scripts/safe_rlhf/sft.sh`.

#### 🐝 Iterative data generation and annotation

Generation for DPO model:
```bash
bash scripts/safe_rlhf/gen_dpo.sh ${iter_number} ${#responses per prompt} # the response number is set as 2 in our experiments
```

Generation for RL model:
```bash
bash scripts/safe_rlhf/gen_pg.sh ${iter_number} ${#responses per prompt} # the response number is set as 2 in our experiments; when iter_number=1, it is equivalent to DPO generation.
```

Annotation for DPO data:
```bash
bash scripts/annotate.sh ${iter_number} dpo ${responses per prompt} # the response number is set as 2 in our experiments
```

We provide a weak (mis-specified) reward model `zhezi12138/weak_rm_gpt2-large_harmless` on huggingface to simulate reward mis-specification.

Annotation for RL data:
```bash
bash scripts/annotate.sh ${iter_number} pg ${responses per prompt} # the response number is set as 2 in our experiments
bash scripts/annotate_weak.sh ${iter_number} pg ${responses per prompt} # annotation with reward model mis-specification
```

#### 🐧 Iterative training

For online DPO training:
```bash
conda activate rlhflow
bash scripts/safe_rlhf/dpo_online.sh ${iter_number} # no policy model mis-specification
bash scripts/safe_rlhf/dpo_online_mis.sh ${iter_number} # with policy model mis-specification
```

For RL training:
```bash
conda activate rlhflow
bash scripts/safe_rlhf/pg.sh ${iter_number} # no policy model mis-specification
bash scripts/safe_rlhf/pg_mis.sh ${iter_number} # with policy model mis-specification
```

#### 🐤 Evaluation
To test the DPO/RL model, the command is:
```bash
bash scripts/safe_rlhf/gen_test.sh 0 ${here `dpo` or `pg`} ${iter_number}
bash scripts/annotate_test.sh 0 ${here `dpo` or `pg`} ${iter_number}
```

Samples of iterative training pipeline are provided as `scirpts/pipeline_train.sh` and `scripts/pipeline_train_mis.sh`, and a sample of evaluation is provided as `scripts/pipeline_test.sh`. 

## Exp-1: Verifications of Section 4

```bash
cd Exp-1
```

### 🔨 Set up
This part is directly based on [modpo](https://github.com/ZHZisZZ/modpo). 

```bash
create -n rml python=3.10
conda activate rml
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 🏄 Pipeline

#### 🍎 DPO

```bash
bash scripts/dpo/run.sh ${data size} ${seed number}
```
In our paper, we adopt data size as 1000,2000,4000,9000, and seed number as 41,42,43.

#### 🍏 RM
 
```bash
bash scripts/rm/run.sh ${data size} ${seed number}
```
In our paper, we adopt data size as 1000,2000,4000,9000, and seed number as 41,42,43.

*Note that the default preference objective is 'better'. And you can change to 'safer' by modifying the 3rd line of the scripts.*

And finally you can compare the evaluation accuracy which would be reported once training terminates.

## 🏷️ License
This repo is licensed under the MIT license. See the LICENSE file for details.

## 📝 Citation
If you find our work useful, please consider citing:

```
@article{shi2025understandingperformancegappreference,
      title={Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO}, 
      author={Ruizhe Shi and Minhak Song and Runlong Zhou and Zihan Zhang and Maryam Fazel and Simon S. Du},
      year={2025},
      journal={arXiv},
      url={https://arxiv.org/abs/2505.19770}, 
}
```
