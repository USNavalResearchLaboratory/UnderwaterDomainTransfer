# To run

python train.py --dataroot ./datasets/hands_mixed_B --name Underwaterizer_mixed --CUT_mode CUT --n_epochs 300 --no_flip --preprocess crop


Running visdom server!
python -m visdom.server



cycle gan

python train.py --dataroot ./datasets/hands_mixed_B --name Underwaterizer_mixed_cycle --model cycle_gan --n_epochs 300 --no_flip --preprocess crop


python train.py --dataroot ./datasets/single_image_hand --name SinCut --model sincut --no_flip --preprocess crop



CUT VAROS

python train.py --dataroot ./datasets/VAROS --name VAROS_Underwaterizer --CUT_mode CUT --n_epochs 300 --no_flip --preprocess crop


CUT VAROS depth

python train.py --dataroot ./datasets/VAROS_depth --name VAROS_Underwaterizer_W_DEPTH --CUT_mode CUT --n_epochs 300 --no_flip --preprocess crop --dataset_mode depth_unaligned --input_nc 4