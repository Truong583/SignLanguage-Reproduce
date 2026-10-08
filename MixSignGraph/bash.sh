 
accelerate launch --config_file local_config.yaml main.py --device 0,1,2 --config ./configs/baseline.yaml -- work_dir ./work_dir/phoenix2014t/s2g/

accelerate launch --config_file local_config.yaml main.py\
 --device 0,1,2 --config ./configs/baseline_slt.yaml\
 -- work_dir ./work_dir/phoenix2014t/s2t/ --load_weights ./work_dir/phoenix2014t/s2g/_best_model.pt
