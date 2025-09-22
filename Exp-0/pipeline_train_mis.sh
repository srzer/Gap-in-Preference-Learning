. "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate rlhflow
bash scripts/safe_rlhf/gen_dpo.sh 1 2
bash scripts/annotate.sh 1 dpo 2
bash scripts/safe_rlhf/dpo_online_mis.sh 1
bash scripts/safe_rlhf/gen_dpo.sh 2 2
bash scripts/annotate.sh 2 dpo 2
bash scripts/safe_rlhf/dpo_online_mis.sh 2
bash scripts/safe_rlhf/gen_dpo.sh 3 2
bash scripts/annotate.sh 3 dpo 2
bash scripts/safe_rlhf/dpo_online_mis.sh 3
bash scripts/safe_rlhf/gen_dpo.sh 4 2
bash scripts/annotate.sh 4 dpo 2
bash scripts/safe_rlhf/dpo_online_mis.sh 4

bash scripts/safe_rlhf/pg_mis.sh 1
bash scripts/safe_rlhf/gen_pg.sh 2 2
bash scripts/annotate.sh 2 pg 2
bash scripts/safe_rlhf/pg_mis.sh 2
bash scripts/safe_rlhf/gen_pg.sh 3 2
bash scripts/annotate.sh 3 pg 2
bash scripts/safe_rlhf/pg_mis.sh 3
bash scripts/safe_rlhf/gen_pg.sh 4 2
bash scripts/annotate.sh 4 pg 2
bash scripts/safe_rlhf/pg_mis.sh 4