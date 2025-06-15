# Understanding the Performance Gap in Preference Learning
This repo provides code for the paper ["Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO"](https://arxiv.org/pdf/2505.19770). By now, we have only released the code for *verifications of Section 4*. And for *verifications of Section 3*, you could refer to the codebase of [Online-RLHF](https://github.com/RLHFlow/Online-RLHF). ***If you find any issue in reproduction, feel free to create an issue!***

## 🔨 Set up
This part is directly based on [modpo](https://github.com/ZHZisZZ/modpo). 

```bash
create -n rml python=3.10
conda activate rml
pip install -r requirements.txt
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

## 🏄 Pipeline

### 🍎 DPO

```bash
bash scripts/dpo/run.sh ${data size} ${seed number}
```
In our paper, we adopt data size as 1000,2000,4000,9000, and seed number as 41,42,43.

### 🍏 RM

```bash
bash scripts/rm/run.sh ${data size} ${seed number}
```
In our paper, we adopt data size as 1000,2000,4000,9000, and seed number as 41,42,43.

*Note that the default preference objective is 'better'. And you can change to 'safer' by modifying the 3rd line of the scripts.*

And finally you can compare the evaluation accuracy which would be reported once training terminates.

## 🏷️ License
This repo is licensed under the MIT license. See the LICENSE file for details.