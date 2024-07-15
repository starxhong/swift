# Experimental environment: A10, 3090, V100
# 20GB GPU memory

## 环境配置
# :<<'EOF'
pip install -e '.[llm]'
cd /cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL
pip install -r requirements.txt # 如果遇到库不兼容的问题，按指引安装指定版本即可
pip install -e .
cd -
pip install deepspeed
pip install -U accelerate
# EOF

dataset=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/eval_data/sharegpt/20240417_evalset_with_image_caption_by_deepseek_pmt_v41_tail512.jsonl
ckpt_dir=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/deepseek-vl-7b-chat/v49-20240511-121553/checkpoint-4260
# ckpt_dir=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/deepseek-vl-7b-chat/v47-20240506-102434/checkpoint-3050
local_repo_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL
result_dir=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/tmp

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model_type deepseek-vl-7b-chat \
    --custom_val_dataset_path ${dataset} \
    --ckpt_dir ${ckpt_dir} \
    --local_repo_path ${local_repo_path} \
    --max_length 4096 \
    --val_dataset_sample -1 \
    --truncation_strategy delete \
    --show_dataset_sample -1 \
    --result_dir ${result_dir}
