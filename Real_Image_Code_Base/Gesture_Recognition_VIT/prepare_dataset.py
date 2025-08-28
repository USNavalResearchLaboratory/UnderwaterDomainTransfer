
import os
import shutil
import re
import json

NAME = 'VAROS_depth_v1_epoch_40'
DESC = 'The 40 epoch CUT gan trained on varos dataset with depth pro depth prediction'
BASE_DIR = './datasets/VAROS_depth_v1_epoch_40'



def sort_data(target_path):
    # Sort any unsorted items in train
    for item in os.listdir(target_path):
        if not os.path.isdir(f'{target_path}/{item}'):
            class_label = re.search('_([a-z]*)', item).group(1).lower()
            shutil.move(f'{target_path}/{item}', f'{target_path}/{class_label}/{item}')

def get_item_counts(target_path: str) -> tuple[dict[any], int]:
    data = dict()
    total = 0
    for item in os.listdir(target_path):
        item_path = f'{target_path}/{item}'
        if os.path.isdir(item_path):
            item_data, amnt_items = get_item_counts(item_path)
            total += amnt_items
            data[item] = {
                'count': amnt_items,
                'includes': item_data
            }
        else:
            total += 1

    return data, total


def build_info_file(target_path, desc, name):

    data = dict()
    data['_meta'] = {
        '_name': name,
        '_desc': desc,
        '_path':target_path
    }
    sub_data, counts = get_item_counts(target_path)
    data['data'] = {
        'total_count': counts,
        'includes': sub_data
    }
    with open(f'{target_path}/info.json', "w") as json_file:
        json.dump(data, json_file, indent=1)





paths = []

# test dirs
os.makedirs(f'{BASE_DIR}/test/open', exist_ok=True)
os.makedirs(f'{BASE_DIR}/test/close', exist_ok=True)
os.makedirs(f'{BASE_DIR}/test/up', exist_ok=True)
os.makedirs(f'{BASE_DIR}/test/down', exist_ok=True)
os.makedirs(f'{BASE_DIR}/test/left', exist_ok=True)
os.makedirs(f'{BASE_DIR}/test/right', exist_ok=True)
# Train dirs
os.makedirs(f'{BASE_DIR}/train/open', exist_ok=True)
os.makedirs(f'{BASE_DIR}/train/close', exist_ok=True)
os.makedirs(f'{BASE_DIR}/train/up', exist_ok=True)
os.makedirs(f'{BASE_DIR}/train/down', exist_ok=True)
os.makedirs(f'{BASE_DIR}/train/left', exist_ok=True)
os.makedirs(f'{BASE_DIR}/train/right', exist_ok=True)


# Sort any unsorted images
sort_data(f'{BASE_DIR}/train')
sort_data(f'{BASE_DIR}/test')

build_info_file(BASE_DIR, DESC, NAME)



