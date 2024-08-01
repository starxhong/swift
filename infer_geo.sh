# Experimental environment: A10, 3090, V100
# 20GB GPU memory

## 环境配置
:<<'EOF'
sh /cephfs/group/teg-openrecom-openrc/starxhong/update_gcc.sh
pip install -e '.[llm]'
pip install deepspeed
pip install -U accelerate
EOF

dataset=/cephfs/group/teg-openrecom-openrc/starxhong/ad_geolocation/prompt/part-00144-96e21c9f-a9dd-4318-b907-9154f12aac23-c000_pmt_geo_v1.jsonl
dataset=/cephfs/group/teg-openrecom-openrc/starxhong/ad_geolocation/prompt/
ckpt_dir=/apdcephfs_qy3/share_1603729/jayrjwang/Qwen1.5-7B-Chat/
model_type=qwen1half-7b-chat
result_dir=/cephfs/group/teg-openrecom-openrc/starxhong/ad_geolocation/result

# 判断dataset是否为目录
if [ -d "$dataset" ]; then
    # 遍历目录，查找所有.json和.jsonl文件
    json_files=$(find "$dataset" -type f -name "*.json" -o -name "*.jsonl")
    # 使用for循环遍历所有找到的文件
    for json_file in $json_files; do
        # 将每个文件路径传递给infer命令作为自定义验证数据集
        CUDA_VISIBLE_DEVICES=0 \
        swift infer \
            --model_type ${model_type} \
            --custom_val_dataset_path "$json_file" \
            --ckpt_dir ${ckpt_dir} \
            --model_id_or_path ${ckpt_dir} \
            --max_length 2700 \
            --val_dataset_sample -1 \
            --truncation_strategy delete \
            --dataset_test_ratio 1.0 \
            --show_dataset_sample -1 \
            --result_dir ${result_dir}
    done
else
    # 如果dataset是一个文件，直接使用该文件路径作为custom_val_dataset_path参数
    CUDA_VISIBLE_DEVICES=0 \
    swift infer \
        --model_type ${model_type} \
        --custom_val_dataset_path "$dataset" \
        --ckpt_dir ${ckpt_dir} \
        --model_id_or_path ${ckpt_dir} \
        --max_length 2700 \
        --val_dataset_sample -1 \
        --truncation_strategy delete \
        --dataset_test_ratio 1.0 \
        --show_dataset_sample -1 \
        --result_dir ${result_dir}
fi
