# # ! deprecated 
# python reflow_train.py \
#     --config reflow/configs/train.py \
#     --workdir logs/tmp \
#     --config.device cuda:0

# * main training use
# ACC_CONFIG_FILE="configs/acc_single_default.yaml"
# GPU_IDS="0"
# NGPU=1
# MASTER_PORT=29501
# accelerate launch --config_file $ACC_CONFIG_FILE --main_process_port $MASTER_PORT --num_processes $NGPU --gpu_ids $GPU_IDS reflow_train_ddp.py \
#     --config reflow/configs/train.py \
#     --workdir logs/tmp

# * sample part
python reflow_sample.py \
    --config reflow/configs/sample.py \
    --eval_folder samples/2reflow_AltInit_rk45 \
    --config.device cuda:0 \
    --config.sampling.ckpt_path logs/tmp/checkpoints/score_model_s200009.pth \
    --config.sampling.use_ode_sampler rk45

N_list=(1 2 3 4 5 10 100 1000)
# N_list=(1 2)
for N in ${N_list[*]}
do
    python reflow_sample.py \
        --config reflow/configs/sample.py \
        --eval_folder samples/2reflow_AltInit_euler_s$N \
        --config.device cuda:0 \
        --config.sampling.ckpt_path logs/tmp/checkpoints/score_model_s200009.pth \
        --config.sampling.use_ode_sampler euler \
        --config.sampling.sample_N $N
done