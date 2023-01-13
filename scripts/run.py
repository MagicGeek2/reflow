import os


def get_train_cmd():
    ACC_CONFIG_FILE = "configs/acc_single_default.yaml"
    MASTER_PORT = 29500 + 0  # 默认从 29500 开始，依次递增
    GPU_IDS = "0"
    NGPU = len(GPU_IDS.split(','))
    cmd = f"accelerate launch --config_file {ACC_CONFIG_FILE} --main_process_port {MASTER_PORT} --num_processes {NGPU} --gpu_ids {GPU_IDS} reflow_train_ddp.py \
        --config reflow/configs/train.py \
        --workdir logs/tmp"
    return cmd


def get_sample_cmd():
    cmd = f"python reflow_sample.py \
        --config reflow/configs/sample.py \
        --eval_folder samples/tmp"
    return cmd


if __name__ == "__main__":
    _CMDS_DICT = {
        'train': get_train_cmd,
        'sample': get_sample_cmd,
    }
    init_cmd = 'train'  # ['train', 'sample']
    get_cmd = _CMDS_DICT[init_cmd]
    cmd = get_cmd()
    print(cmd)
    ...
