# python reflow_train.py \
#     --config reflow/configs/main.py \
#     --workdir logs/alt_1step_distillation \
#     --config.device cuda:1

ACC_CONFIG_FILE="configs/acc_multi_default.yaml"
GPU_IDS="1,2"
MASTER_PORT=29501
accelerate launch --config_file $ACC_CONFIG_FILE --main_process_port $MASTER_PORT --gpu_ids $GPU_IDS reflow_train_ddp.py \
    --config reflow/configs/main.py \
    --workdir logs/distill_1step_Alt