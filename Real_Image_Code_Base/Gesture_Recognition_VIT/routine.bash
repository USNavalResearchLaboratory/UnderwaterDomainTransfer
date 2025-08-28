#!/bin/bash 

n_epochs=300
n_patience=300
batch_size=32
source venv/bin/activate
# python train.py --dataset "./datasets/hands_cropped" --name "hands_cropped" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/hands_full_body" --name "hands_full_body" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/hands_underwaterizer_v1_epoch_5" --name "hands_underwaterizer_v1_epoch_5" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/hands_underwaterizer_v1_epoch_10" --name "hands_underwaterizer_v1_epoch_10" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/hands_underwaterizer_v1_epoch_15" --name "hands_underwaterizer_v1_epoch_15" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/hands_underwaterizer_v1_epoch_15_CHEAT" --name "hands_underwaterizer_v1_epoch_15_CHEAT" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/cycle_v1_epoch_10" --name "cycle_v1_epoch_10" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/cycle_v1_epoch_25" --name "cycle_v1_epoch_25" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_v1_epoch_35" --name "VAROS_v1_epoch_35" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_v1_epoch_45" --name "VAROS_v1_epoch_45" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_v1_epoch_55" --name "VAROS_v1_epoch_55" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_v1_epoch_65" --name "VAROS_v1_epoch_65" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_depth_v1_epoch_5" --name "VAROS_depth_v1_epoch_5" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_depth_v1_epoch_30" --name "VAROS_depth_v1_epoch_30" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train.py --dataset "./datasets/VAROS_depth_v1_epoch_40" --name "VAROS_depth_v1_epoch_40" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size


# python train_res_net.py --dataset "./datasets/hands_cropped" --name "res18_hands_cropped" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/hands_underwaterizer_v1_epoch_5" --name "res18_hands_underwaterizer_v1_epoch_5" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/hands_underwaterizer_v1_epoch_10" --name "res18_hands_underwaterizer_v1_epoch_10" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/hands_underwaterizer_v1_epoch_15" --name "res18_hands_underwaterizer_v1_epoch_15" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/VAROS_v1_epoch_35" --name "res18_VAROS_v1_epoch_35" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/VAROS_v1_epoch_45" --name "res18_VAROS_v1_epoch_45" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/VAROS_v1_epoch_55" --name "res18_VAROS_v1_epoch_55" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/VAROS_v1_epoch_65" --name "res18_VAROS_v1_epoch_65" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
python train_res_net.py --dataset "./datasets/VAROS_depth_v1_epoch_5" --name "res18_VAROS_depth_v1_epoch_5" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
python train_res_net.py --dataset "./datasets/VAROS_depth_v1_epoch_30" --name "res18_VAROS_depth_v1_epoch_30" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
python train_res_net.py --dataset "./datasets/VAROS_depth_v1_epoch_40" --name "res18_VAROS_depth_v1_epoch_40" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
# python train_res_net.py --dataset "./datasets/hands_underwaterizer_v1_epoch_15_CHEAT" --name "res18_hands_underwaterizer_v1_epoch_15_CHEAT" --n_epochs $n_epochs --n_patience $n_patience --batch_size $batch_size
