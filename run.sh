# # ! deprecated 
# python reflow_train.py \
#     --config reflow/configs/train.py \
#     --workdir logs/tmp \
#     --config.device cuda:0

# # * main training use
# ACC_CONFIG_FILE="configs/acc_multi_default.yaml"
# GPU_IDS="0,1,2,3,4,5,6,7"
# NGPU=8
# MASTER_PORT=29500
# accelerate launch --config_file $ACC_CONFIG_FILE --main_process_port $MASTER_PORT --num_processes $NGPU --gpu_ids $GPU_IDS reflow_train_ddp.py \
#     --config reflow/configs/train.py \
#     --workdir logs/2_reflow_AltInit_v2 \
#     --comment "1epoch iteration (5M) ; bs= (8gpu x 3 x 1acc) ; linear lr decay"



# # * sample part



# EVAL_FOLDER=samples/distill_1step_Alt
# DEVICE=cuda:2
# randz0=fix
# ckpt_path=logs/2_reflow_AltInit/checkpoints/score_model_s200009.pth
# phase=train

# # python reflow_sample.py \
# #     --config reflow/configs/sample.py \
# #     --eval_folder $EVAL_FOLDER/rk45 \
# #     --config.device $DEVICE \
# #     --config.sampling.use_ode_sampler rk45 \
# #     --config.sampling.randz0 $randz0 \
# #     --config.data.phase $phase
# #     # --config.sampling.ckpt_path $ckpt_path \

# N_list=(1 2 3 4 5 10 100 1000)
# # N_list=(100 1000)
# for N in ${N_list[*]}
# do
#     python reflow_sample.py \
#         --config reflow/configs/sample.py \
#         --eval_folder $EVAL_FOLDER/euler_s$N \
#         --config.device $DEVICE \
#         --config.sampling.use_ode_sampler euler \
#         --config.sampling.sample_N $N \
#         --config.sampling.randz0 $randz0 \
#         --config.sampling.ckpt_path $ckpt_path \
#         --config.data.phase $phase
# done
