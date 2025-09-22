. "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate rlhflow
bash scripts/safe_rlhf/gen_test.sh 0 dpo 1
bash scripts/safe_rlhf/gen_test.sh 0 pg 1
bash scripts/safe_rlhf/gen_test.sh 0 dpo 2
bash scripts/safe_rlhf/gen_test.sh 0 pg 2
bash scripts/safe_rlhf/gen_test.sh 0 dpo 3
bash scripts/safe_rlhf/gen_test.sh 0 pg 3
bash scripts/safe_rlhf/gen_test.sh 0 dpo 4
bash scripts/safe_rlhf/gen_test.sh 0 pg 4

bash scripts/annotate_test.sh 0 dpo 1
bash scripts/annotate_test.sh 0 pg 1
bash scripts/annotate_test.sh 0 dpo 2
bash scripts/annotate_test.sh 0 pg 2
bash scripts/annotate_test.sh 0 dpo 3
bash scripts/annotate_test.sh 0 pg 3
bash scripts/annotate_test.sh 0 dpo 4
bash scripts/annotate_test.sh 0 pg 4