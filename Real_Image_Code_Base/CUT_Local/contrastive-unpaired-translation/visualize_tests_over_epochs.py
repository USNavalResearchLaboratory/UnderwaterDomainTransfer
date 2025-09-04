
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from math import ceil, sqrt


MODEL_NAME = 'VAROS_Underwaterizer_W_DEPTH'
BASE_PATH = f'./results/{MODEL_NAME}'
IMAGE_INDEXS = [0, 1, 2, 3]
print('Working in:', os.getcwd())
for image_index in IMAGE_INDEXS:

    print('Working on index:', image_index)


    # Gather image paths
    images = []
    input_image_path = None

    for test in os.scandir(BASE_PATH):
        print("Working on test:", test.name)
        epoch_num = int(test.name[5:])

        # Save input image that is the same over all the tests
        if input_image_path is None:
            image_dir_path = f'{BASE_PATH}/{test.name}/images/real_A'
            input_image_path = list(os.scandir(image_dir_path))[image_index]

        
        # Grab the correct output image
        image_dir_path = f'{BASE_PATH}/{test.name}/images/fake_B'
        test_image = list(os.scandir(image_dir_path))[image_index]
        images.append((epoch_num, test_image))


    sorted_by_epoch_images = sorted(images, key=lambda x: x[0])

    num_images = len(sorted_by_epoch_images) + 1 # plus 1 to include input image
    x = ceil(sqrt(num_images))
    fig, axes = plt.subplots(x, x, figsize=(8, 8)) # Create a 2x2 grid of subplots
    axes = axes.flatten() # Flatten the axes array for easier indexing

    for i, image in enumerate(sorted_by_epoch_images):
        print('Reading:', image[1])
        img = mpimg.imread(image[1])
    
        axes[i].imshow(img)
        axes[i].text(0.95, 0.95, image[0], transform=axes[i].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='red')
        axes[i].axis('off')  

    for j, ax in enumerate(axes):

        if j > i:
            ax.set_visible(False)

    # Show input image
    img = mpimg.imread(input_image_path)
    last_axes = axes[-1]
    last_axes.imshow(img)
    last_axes.text(0.95, 0.95, 'input', transform=last_axes.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='red')
    last_axes.axis('off')  
    last_axes.set_visible(True)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f'./results/PostProcessed/{MODEL_NAME}_{image_index}.png', dpi=300)
