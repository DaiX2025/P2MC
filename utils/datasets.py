import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])
mask_valid_array = np.array([[False, False, True, False],
            [False, True, True, False],
            [True, True, False, True],
            [True, True, True, True]])

class SegDataset(Dataset):
    def __init__(self, data_dir, data_file, transform_trn=None, transform_val=None, transform_test=None, stage='train', num_cls=4):
        with open(data_file, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        self.datanames = datalist

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(data_dir, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths

        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.stage = stage
        self.num_cls = num_cls


    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.volpaths)
    
    def __getitem__(self, index):

        volpath = self.volpaths[index]
        dataname = self.datanames[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...] #[1,H,W,D,C] [1,H,W,D]

        if self.stage == 'train':
            if self.transform_trn:
                x,y = self.transform_trn([x, y])
        elif self.stage == 'val':
            if self.transform_val:
                x,y = self.transform_val([x, y])
        elif self.stage == 'test':
            if self.transform_test:
                x,y = self.transform_test([x, y])   

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) #[1,C,H,W,D]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))

        # 将标签值裁剪到有效范围内
        y = np.clip(y, 0, self.num_cls - 1)
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3)) #[1,num_cls,H,W,D]

        x = torch.squeeze(torch.from_numpy(x), dim=0) #[C,H,W,D]
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) #[num_cls,H,W,D]
        
        if self.stage == 'train':
            mask_idx = np.random.choice(15, 1)
            mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #[C]
            # mask = torch.squeeze(torch.from_numpy(np.array([True, True, True, True])), dim=0)
            return x, yo, mask, dataname
        else:
            return x, yo, dataname

    
class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)