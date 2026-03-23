import os
import medpy.io as medio
import numpy as np

src_path = ''
tar_path = ''

name_list = os.listdir(src_path)

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

modalities = ['t1ce', 't2', 't2fs', 'dwi', 'seg', 'vol']
for modality in modalities:
    if not os.path.exists(os.path.join(tar_path, modality)):
        os.makedirs(os.path.join(tar_path, modality))

for file_name in name_list:
    print (file_name)
    t1ce, t1ce_header = medio.load(os.path.join(src_path, file_name, file_name+'_t1ce.nii.gz'))
    t2, t2_header = medio.load(os.path.join(src_path, file_name, file_name+'_t2.nii.gz'))
    t2fs, t2fs_header = medio.load(os.path.join(src_path, file_name, file_name+'_t2fs.nii.gz'))
    dwi, dwi_header = medio.load(os.path.join(src_path, file_name, file_name+'_dwi.nii.gz'))

    vol = np.stack((t1ce, t2, t2fs, dwi), axis=0).astype(np.float32)
    print(vol.shape)
    vol1 = normalize(vol)
    print(np.min(vol1), np.max(vol1))
    print(vol1.shape)

    np.save(os.path.join(tar_path, 't1ce', file_name+'_t1ce.npy'), vol1[0])
    np.save(os.path.join(tar_path, 't2', file_name+'_t2.npy'), vol1[1])
    np.save(os.path.join(tar_path, 't2fs', file_name+'_t2fs.npy'), vol1[2])
    np.save(os.path.join(tar_path, 'dwi', file_name+'_dwi.npy'), vol1[3])

    vol1 = vol1.transpose(1,2,3,0)
    print(vol1.shape)
    np.save(os.path.join(tar_path, 'vol', file_name+'_vol.npy'), vol1)

    seg, seg_header = medio.load(os.path.join(src_path, file_name, file_name+'_seg.nii.gz'))
    seg = seg.astype(np.uint8)
    print(seg.shape)
    print(np.unique(seg))

    np.save(os.path.join(tar_path, 'seg', file_name+'_seg.npy'), seg)
