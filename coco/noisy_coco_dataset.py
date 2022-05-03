#
# Uber, Inc. (c) 2017
#
# COCO data sets.
#
from __future__ import absolute_import, division, print_function

from datasets.base.dataset import Dataset
from datasets.data_factory import RegisterDataset


@RegisterDataset("coco-noisy")
class NoisyCOCODataset(Dataset):
    """COCO data set."""

    def __init__(self, subset, data_dir, num_clean, num_val):
        super(NoisyCOCODataset, self).__init__('Noisy COCO', subset, data_dir)
        self._num_clean = num_clean
        self._num_val = num_val

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train_noisy':
            return 20000 - self.num_val - self.num_clean
        elif self.subset == 'train_clean':
            return self.num_clean
        elif self.subset in ['validation', 'validation_noisy']:
            return self.num_val
        elif self.subset == 'test':
            return 20000

    def download_message(self):
        pass

    def available_subsets(self):
        return ['train_noisy', 'train_clean', 'validation', 'validation_noisy', 'test']

    @property
    def num_clean(self):
        return self._num_clean

    @property
    def num_val(self):
        return self._num_val
    
    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10

