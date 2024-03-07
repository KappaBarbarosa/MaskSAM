export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
rm -rf results/train/*
rm -rf results/valid/*
python train.py \
    --data_config 'Configs/data_config.yml' \
    --model_config 'Configs/model_config.yml' \
    --valid_config 'Synapse/val_list.json' \
    --save_path 'checkpoints' \
    --img_save_path 'results1'\
    --device 'cuda:0' \
