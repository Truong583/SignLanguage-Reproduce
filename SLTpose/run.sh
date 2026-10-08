





#accelerate launch --config_file local_config.yaml mainTrain.py \
#--device 0,1,2 \
#--config ./configs/baseline.yaml \
#--phase train \
#--wandb_name 'phoenix14T' \
#--load-checkpoints './work_dir/Phoenix14T/SLT_zuang/_best_model.pt' \
#--work-dir './work_dir/Phoenix14T/SLT_zuang/'


accelerate launch --config_file local_config.yaml mainTrain.py \
--device 0,1,2 \
--config ./configs/baseline_slt.yaml \
--phase train \
--wandb_name 'phoenix14T' \
--load-weight './work_dir/Phoenix14T/SLT_zuang/_best_model.pt' \
--work-dir './work_dir/Phoenix14T/SLT_zuang/'