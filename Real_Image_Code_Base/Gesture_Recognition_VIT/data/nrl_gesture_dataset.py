from data.base_dataset import BaseDataset, get_transform
import os
import re
from PIL import Image

class NRLGestureDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.CLASS_MAP = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3,
            'open': 4,
            'close': 5
        }
        self._load_set_via_NRL_names()
        self._data_len = len(self.x)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self._data_len

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """

        _path = self.x[index % self._data_len]  
        _cls = self.y[index % self._data_len]

        
        x_img = Image.open(_path).convert('RGB')
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(self.opt)
        x = transform(x_img)

        return {'X': x, 'Y': _cls}



    def _pull_x_image_paths(self, dir, max_dataset_size=float("inf")):
        paths = []
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                paths.append(f'{root}/{fname}')

                if paths.__len__() >= max_dataset_size:
                    break

        return paths

    def _get_class_labels_from_paths(self, paths: list[str]) -> list[int]:
        # Format 1: Ed_close0.png
        # Format 2: underwater_close_4446.png
        # Returns class label in group 1: '_([a-z]*)'
        labels = []
        for path in paths:
            class_label = re.search('_([a-z]*)', path).group(1).lower()
            cls_idx = self.CLASS_MAP.get(class_label)
            if cls_idx is None:
                raise IndexError(f'Class map does not contain class label {class_label}, ABORT')
            labels.append(cls_idx)

        return labels




    
    def _load_set_via_NRL_names(self):
        root_path = f'{self.root}/{'trainX' if self.opt.isTrain else 'testX'}'
        self.x = self._pull_x_image_paths(root_path)
        self.y = self._get_class_labels_from_paths(self.x)