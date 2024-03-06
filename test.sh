rm -rf ./test_results/*
# rm -rf ./test_results_1/*
python test_synapse.py \
    --data_config 'configs/data_config.yml' \
    --model_config 'configs/model_config.yml' \
    --pretrained_path '/home/kappa7077/MaskSam/checkpoints/03-04_17:11' \
    --batch_size 75 \
    --device 'cuda:0'
# python test.py \
#     --data_config 'configs/Instrument_config.yml' \
#     --model_config 'configs/model_config.yml' \
#     --pretrained_path '/home/kappa7077/MaskSam/checkpoints/03-06_15:49' \
#     --batch_size 500 \
#     --device 'cuda:0'