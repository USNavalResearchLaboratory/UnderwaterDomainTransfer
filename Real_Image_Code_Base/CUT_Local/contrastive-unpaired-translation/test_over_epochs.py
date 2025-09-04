import subprocess


FIRST_EPOCH = 5
LAST_EPOCH = 70
MODEL_NAME = 'VAROS_Underwaterizer_W_DEPTH'
DATASET = 'datasets/VAROS_depth/'
MODEL = 'cut'


for current_epoch in range(FIRST_EPOCH, LAST_EPOCH, 5):
    print(f'======================== {current_epoch}/{LAST_EPOCH} ========================')
        
    # Run testing over each epoch (every 5 are saved in the checkpoint)
    test_script_command = ['python', 'test.py', '--dataroot', DATASET, '--name', MODEL_NAME, '--model', MODEL, '--epoch', str(current_epoch), '--input_nc',  '4', '--output_nc', '4', '--dataset_mode', 'depth_unaligned']


    subprocess.call(test_script_command)



# python test.py --dataroot datasets/hands/ --name Underwaterizer_v1 --model cut --epoch 5 --dataset_mode class_aligned

# python test.py --dataroot datasets/cropped_mixed/ --name Underwaterizer_mixed_cycle --model cycle_gan --epoch 10
# python test.py --dataroot datasets/single_image_hand/ --name SinCut --model sincut --epoch 3



# python test.py --dataroot datasets/VAROS/ --name VAROS_Underwaterizer --model cut --epoch 65


# python test.py --dataroot datasets/VAROS_depth/ --name VAROS_Underwaterizer_W_DEPTH --model cut --epoch 65 --nc-input 4 --nc-output 4


# Varos with depth 
# python test.py --dataroot ./datasets/VAROS_depth/ --name VAROS_Underwaterizer_W_DEPTH --model cut --epoch 65 --input_nc 4 --output_nc 4 --dataset_mode depth_unaligned