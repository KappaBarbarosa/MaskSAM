rm -rf ./test_results/*
# rm -rf ./test_results_1/*
# python test_synapse.py \
#     --data_config 'configs/data_config.yml' \
#     --model_config 'configs/model_config.yml' \
#     --pretrained_path '/home/kappa7077/MaskSam/checkpoints/03-04_17:11' \
#     --batch_size 75 \
#     --device 'cuda:0'
python test_others.py \
    --data_config 'Configs/CheXdet_config.yml' \
    --model_config 'Configs/model_config.yml' \
    --pretrained_path '/home/kappa7077/MaskSam/checkpoints/03-06_02:00' \
    --batch_size 500 \
    --device 'cuda:0'