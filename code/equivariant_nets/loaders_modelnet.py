import os
import glob
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, ConcatDataset


class PreSampledDataset(torch.utils.data.Dataset):
    def __init__(self, mode, root, div, epoch=0, epoch_mod=30, unique_samplings='auto', dtype=np.float32):
        self.root = os.path.join(os.path.realpath(os.path.expanduser(root)), mode)
        self.mode = mode
        self.div = np.asarray(div)
        self.unique_samplings = mode not in ['test', 'valid'] if unique_samplings is None else unique_samplings
        self.dtype = dtype
        self.path_ep = lambda ep: os.path.join(self.root, f'ep{ep}')
        self.paths = sorted([os.path.relpath(p, self.path_ep(0)) for p in glob.glob(os.path.join(self.path_ep(0), '*', '*.npy'))])
        self.label_str = [f.split(os.path.sep)[0] for f in self.paths]
        self.label_map = sorted(set(self.label_str))
        self.label_num = list(map(self.label_map.index, self.label_str))
        self.n_classes = len(self.label_map)
        self.epoch = epoch
        self.epoch_mod = epoch_mod

    def __len__(self):
        return len(self.label_num)

    def next_epoch(self):
        self.epoch = (self.epoch + bool(self.unique_samplings)) % self.epoch_mod

    @property
    def items_per_class(self):
        i, n = np.unique(self.label_num, return_counts=True)
        result = np.zeros(self.n_classes, dtype=np.int64)
        result[i] = n
        return result

    @property
    def items_per_class_with_names(self):
        return [(self.label_map[i], n) for i, n in zip(*np.unique(self.label_num, return_counts=True))]

    def __getitem__(self, idx):
        data_file = os.path.join(self.path_ep(self.epoch), self.paths[idx])
        return np.load(data_file), self.label_num[idx]


    def collate(self, batch):
        data, *rest = list(zip(*batch))
        data_cat = torch.cat(list(map(torch.as_tensor, data)))
        obj_size = (torch.tensor(list(map(len, data))) - 1) // torch.as_tensor(self.div)[:, None] + 1
        return [(*data_cat.movedim(1, 0), obj_size), *map(default_collate, rest)]


class PreSampledModelNet40(PreSampledDataset):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/modelnet40_sampled', div=4**np.arange(4), **kwargs)
        assert self.items_per_class_with_names == {  # check if number of items match
            'train': [('airplane', 626), ('bathtub', 106), ('bed', 515), ('bench', 173), ('bookshelf', 572), ('bottle', 335), ('bowl', 64), ('car', 197), ('chair', 889), ('cone', 167), ('cup', 79), ('curtain', 138), ('desk', 200), ('door', 109), ('dresser', 200), ('flowerpot', 149), ('glassbox', 171), ('guitar', 155), ('keyboard', 145), ('lamp', 124), ('laptop', 149), ('mantel', 284), ('monitor', 465), ('nightstand', 200), ('person', 88), ('piano', 231), ('plant', 240), ('radio', 104), ('rangehood', 115), ('sink', 128), ('sofa', 680), ('stairs', 124), ('stool', 90), ('table', 392), ('tent', 163), ('toilet', 344), ('tvstand', 267), ('vase', 475), ('wardrobe', 87), ('xbox', 103)],
            'test':  [('airplane', 100), ('bathtub', 50), ('bed', 100), ('bench', 20), ('bookshelf', 100), ('bottle', 100), ('bowl', 20), ('car', 100), ('chair', 100), ('cone', 20), ('cup', 20), ('curtain', 20), ('desk', 86), ('door', 20), ('dresser', 86), ('flowerpot', 20), ('glassbox', 100), ('guitar', 100), ('keyboard', 20), ('lamp', 20), ('laptop', 20), ('mantel', 100), ('monitor', 100), ('nightstand', 86), ('person', 20), ('piano', 100), ('plant', 100), ('radio', 20), ('rangehood', 100), ('sink', 20), ('sofa', 100), ('stairs', 20), ('stool', 20), ('table', 100), ('tent', 20), ('toilet', 100), ('tvstand', 100), ('vase', 100), ('wardrobe', 20), ('xbox', 20)],
        }[self.mode]


def loaders_modelnet40_presampled(batch_size, test_samplings=4, test_shuffle=False, **kwargs):
    train_dataset = PreSampledModelNet40(mode='train')
    test_dataset = ConcatDataset([PreSampledModelNet40(mode='test', epoch=epoch) for epoch in range(test_samplings)])
    params = dict(batch_size=batch_size, **{'num_workers': 8, 'pin_memory': True, **kwargs})
    return {
        'train': DataLoader(train_dataset, collate_fn=train_dataset.collate, shuffle=True, **params),
        'test':  DataLoader(test_dataset, collate_fn=test_dataset.datasets[0].collate, shuffle=test_shuffle, **params),
    }
