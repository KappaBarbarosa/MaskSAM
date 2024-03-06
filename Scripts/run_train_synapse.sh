export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
rm -rf ./results/train/*
rm -rf ./results/valid/*
python driver_scratchpad.py \
    --data_config 'configs/data_config.yml' \
    --model_config 'configs/model_config.yml' \
    --valid_config 'val_list.json' \
    --save_path './checkpoints' \
    --img_save_path './results1'\
    --device 'cuda:0' \
