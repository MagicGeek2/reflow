# python reflow_train.py \
#     --config reflow/configs/main.py \
#     --workdir logs/alt_1step_distillation \
#     --config.device cuda:1

ACC_CONFIG_FILE="configs/acc_multi_default.yaml"
GPU_IDS="2,3"
MASTER_PORT=29500
accelerate launch --config_file $ACC_CONFIG_FILE --main_process_port $MASTER_PORT --gpu_ids $GPU_IDS reflow_train_multigpu.py \
    --config reflow/configs/main.py \
    --workdir logs/tmp