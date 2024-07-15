checkpoint=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/deepseek-vl-7b-chat/v39-20240428-234401/checkpoint-3090
local_repo_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir ${checkpoint} \
    --merge_lora true \
    --local_repo_path ${local_repo_path} \
    --model_type deepseek-vl-7b-chat \
