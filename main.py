# general libs
import os, sys, argparse, re
import random, time
import warnings

warnings.filterwarnings('ignore')
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from config import *
from utils.helpers import *
from utils.meter import *
import utils.helpers as helpers
from utils import criterions

from utils.datasets import SegDataset as Dataset
from utils.datasets import MultiEpochsDataLoader
from utils.data_utils import init_fn
from utils.transforms import *
from utils.optimizer import PolyWarmupAdamW


from models import mcbtformer
from predict import test_dice_hd95_softmax, test_dice_hd95_softmax_cc

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset name')
    parser.add_argument('-d', '--train-dir', type=str, default=TRAIN_DIR,
                        help='Path to the training set directory.')
    parser.add_argument('--val-dir', type=str, default=VAL_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                        help='Path to the test set directory.')
    parser.add_argument('--train-list', type=str, default=TRAIN_LIST,
                        help='Path to the training set list.')
    parser.add_argument('--val-list', type=str, default=VAL_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--test-list', type=str, default=TEST_LIST,
                        help='Path to the test set list.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE,
                        help='Crop size for training,')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size to train the segmenter model.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes for each task.')

    
    # General
    parser.add_argument('--model', type=str, default=MODEL,
                        help='model name')
    parser.add_argument('--pretrained', type=bool, default=PRETRAINED,
                        help='Whether to init with pretrained weights.')
    parser.add_argument('--pretrained-path', type=str, default=PRETRAINEDPATH,
                        help='path to pretrained model')
    parser.add_argument('--alpha', type=int, default=ALPHA,
                        help='weight of alignment loss.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1],
                        help='select gpu.')
    parser.add_argument('--validate', type=bool, default=NEEDVAL,
                        help='If true, validate best epoch while training.')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If true, only test.')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('-s', '--save-dir', type=str, metavar='PATH', default='exps',
                        help='path to save checkpoint for different expriments(default: exps)')
    parser.add_argument('--resume', type=str, metavar='PATH', default=RESUME,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-every', type=int, default=VAL_EVERY,
                        help='How often to validate current architecture.')
    parser.add_argument('--save-every', type=int, default=500,
                        help='How often to savee current architecture.')
    parser.add_argument('--print-network', action='store_true', default=False,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--print-loss', action='store_true', default=False,
                        help='Whether print losses during training.')
    parser.add_argument('--save-image', action='store_true', default=False,
                        help='whether to save images during evaluating.')
    parser.add_argument('-i', '--input', default=MODALITY, type=str, nargs='+', 
                        help='input type (falir, t1c, t1, t2)')

    # Optimisers
    parser.add_argument('--num-epochs', type=int, default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument('--iter', type=int, default=ITER_PER_EPOCH,
                        help='Number of iters to train for each epoch.')
    parser.add_argument('--rfse', type=int, default=RFSE,
                        help='Region fusion start epoch.')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate.')
    parser.add_argument('--warmup', type=int, default=WARMUP,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=WD,
                        help='Weight decay.')
    return parser.parse_args()


def create_segmenter(model_name, img_size, num_cls, gpu):
    if model_name == 'mcbtformer':
        segmenter = mcbtformer.Model(img_size=img_size, num_classes=num_cls)
        param_groups = segmenter.parameters()
    else:
        print_log(f"Error: Model '{model_name}' is not recognized.")
        exit(0)
    
    assert(torch.cuda.is_available())
    segmenter.to(gpu[0])
    segmenter = torch.nn.DataParallel(segmenter, gpu)
    return segmenter, param_groups


def create_loaders(train_dir, val_dir, test_dir, train_list, val_list, test_list, crop_size, batch_size, num_workers, num_cls):
    transform_trn = Compose([
        RandCrop3D(crop_size), 
        RandomRotion(10), 
        RandomIntensityChange((0.1,0.1)), 
        RandomFlip(0), 
        NumpyType((np.float32, np.int64)),])
    transform_val = Compose([ 
        NumpyType((np.float32, np.int64)),])
    transform_test = Compose([
        NumpyType((np.float32, np.int64)),])
    
    # Training and validation datasets
    trainset = Dataset(data_dir=train_dir, data_file=train_list, transform_trn=transform_trn, transform_val=None, transform_test=None, stage='train', num_cls=num_cls)
    validset = Dataset(data_dir=val_dir, data_file=val_list, transform_trn=None, transform_val=transform_val, transform_test=None, stage='val', num_cls=num_cls)
    testset = Dataset(data_dir=test_dir, data_file=test_list, transform_trn=None, transform_val=None, transform_test=transform_test, stage='test', num_cls=num_cls)
        
    print_log('Created train set {} examples, val set {} examples, test set {} examples'.format(len(trainset), len(validset), len(testset)))
    # Training, validation and test dataloaders
    train_loader = MultiEpochsDataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, worker_init_fn=init_fn)
    val_loader = MultiEpochsDataLoader(dataset=validset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def load_ckpt(ckpt_path, ckpt_dict, load_optimizer=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    for k, v in ckpt_dict.items():
        if k in ckpt:
            state_dict = ckpt[k]

            if isinstance(v, torch.nn.Module):
                # new_state_dict = {}
                # for param_name, param_value in state_dict.items():
                #     if param_name.startswith("module."):
                #         new_state_dict[param_name[7:]] = param_value
                #     else:
                #         new_state_dict[param_name] = param_value
                # v.load_state_dict(new_state_dict, strict=False)
                
                missing_keys, unexpected_keys = v.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print_log(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print_log(f"Unexpected keys: {unexpected_keys}")


            elif isinstance(v, torch.optim.Optimizer) and load_optimizer:
                try:
                    v.load_state_dict(state_dict)
                except ValueError as e:
                    print_log(f"Skipping optimizer loading for {k} due to mismatch: {e}")

    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    print_log(f'Found checkpoint at {ckpt_path} with best_val {best_val:.4f} at epoch {epoch_start}')
    
    return best_val, epoch_start


def train(segmenter, train_loader, train_iter, optimizer, total_epoch, current_epoch, iter_per_epoch, rfse, num_cls, 
          alpha, print_loss=True):
    train_loader.dataset.set_stage('train')
    segmenter.train()
    train_start = time.time()
    iter_time = AverageMeter()
    losses = AverageMeter()
    for i in range(iter_per_epoch):
        step = (i+1) + current_epoch*iter_per_epoch
        # Load data
        try:
            data = next(train_iter)
        except:
            train_iter = iter(train_loader)
            data = next(train_iter)
        start = time.time()
        inputs, target, mask = data[:3]
        # print_log('train inputs:{} target:{} mask:{}'.format(inputs.shape, target.shape, mask.shape))
        inputs = inputs.float().cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)
        mask = mask.bool().cuda(non_blocking=True)
        
        # Compute loss
        segmenter.module.is_training = True
        fuse_pred, sep_preds, aug_preds, alignment_loss = segmenter(inputs, mask)

        fuse_loss = torch.zeros(1).float().cuda()
        fuse_cross_loss = torch.zeros(1).float().cuda()
        fuse_dice_loss = torch.zeros(1).float().cuda()
        if fuse_pred is not None:
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

        sep_loss = torch.zeros(1).float().cuda()
        sep_cross_loss = torch.zeros(1).float().cuda()
        sep_dice_loss = torch.zeros(1).float().cuda()
        if sep_preds is not None:
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

        # sep_loss = torch.zeros(1).float().cuda()
        # sep_cross_loss = torch.zeros(1).float().cuda()
        # sep_dice_loss = torch.zeros(1).float().cuda()
        # if sep_preds is not None:
        #     B = mask.shape[0]
        #     K = mask.shape[1]
        #     for b in range(B):
        #         for k in range(K):
        #             if mask[b, k]:
        #                 sep_cross_loss += criterions.softmax_weighted_loss(sep_preds[k][b].unsqueeze(0), target[b].unsqueeze(0), num_cls=num_cls)
        #                 sep_dice_loss += criterions.dice_loss(sep_preds[k][b].unsqueeze(0), target[b].unsqueeze(0), num_cls=num_cls)
        #     sep_loss = sep_cross_loss + sep_dice_loss

        aug_loss = torch.zeros(1).float().cuda()
        aug_cross_loss = torch.zeros(1).float().cuda()
        aug_dice_loss = torch.zeros(1).float().cuda()
        if aug_preds is not None:
            for aug_pred in aug_preds:
                aug_cross_loss += criterions.softmax_weighted_loss(aug_pred, target, num_cls=num_cls)
                aug_dice_loss += criterions.dice_loss(aug_pred, target, num_cls=num_cls)
            aug_loss = aug_cross_loss + aug_dice_loss

        alignment_loss_value = torch.zeros(1).float().cuda()
        if alignment_loss is not None:
            alignment_loss_value = alignment_loss.float().cuda()

        if current_epoch < rfse:
            loss = fuse_loss * 0.0 + sep_loss * 0.0 + aug_loss  + alpha * alignment_loss_value
        else:
            loss = fuse_loss + sep_loss + aug_loss  + alpha * alignment_loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), global_step=step)

        if fuse_pred is not None:
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
        
        if sep_preds is not None:
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)

        if aug_preds is not None:
            writer.add_scalar('aug_cross_loss', aug_cross_loss.item(), global_step=step)
            writer.add_scalar('aug_dice_loss', aug_dice_loss.item(), global_step=step)
        
        if alignment_loss is not None:
            writer.add_scalar('alignment_loss', alignment_loss_value.item(), global_step=step)

        if print_loss:
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((current_epoch+1), total_epoch, (i+1), iter_per_epoch, loss.item())

            if fuse_pred is not None:
                msg += 'fusecross:{:.4f}, fusedice:{:.4f}, '.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            # else:
            #     msg += 'fusecross:N/A, fusedice:N/A,'

            if sep_preds is not None:
                msg += 'sepcross:{:.4f}, sepdice:{:.4f}, '.format(sep_cross_loss.item(), sep_dice_loss.item())
            # else:
            #     msg += 'sepcross:N/A, sepdice:N/A,'

            if aug_preds is not None:
                msg += 'augcross:{:.4f}, augdice:{:.4f}, '.format(aug_cross_loss.item(), aug_dice_loss.item())
            # else:
            #     msg += 'augcross:N/A, augdice:N/A,'

            if alignment_loss is not None:
                msg += 'alignment_loss:{:.4f}, '.format(alignment_loss_value.item())
            # else:
            #     msg += 'alignment_loss:N/A,'

            print_log(msg)

        losses.update(loss.item())
        iter_time.update(time.time() - start)
    
    print_log('Train time per iter: {}'.format(iter_time.avg))
    print_log('Train time per epoch: {}'.format(time.time() - train_start))


def validate(segmenter, dataset, val_loader, current_epoch, num_cls, patch_size):
    if dataset in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        mask_name_valid = ['t2', 't1c', 't1', 'flair',
                    't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                    'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                    'flairt1cet1t2']
    elif dataset in ['CC2024']:
        mask_name_valid = ['dwi', 't2', 't2fs', 't1ce',
                    't2dwi', 't2t2fs', 't1cet2fs', 't2fsdwi', 't1cedwi', 't1cet2',
                    't1cet2t2fs', 't1cet2fsdwi', 't1cet2dwi', 't2t2fsdwi',
                    't1cet2t2fsdwi']
    masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
            [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
            [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
            [True, True, True, True]]

    val_loader.dataset.set_stage('val')
    segmenter.eval()
    val_start = time.time()
    batch_time = AverageMeter()
    val_dice_score = AverageMeter()
    dice_modality = torch.zeros(15)
    patch_size_h, patch_size_w, patch_size_z = patch_size
    one_tensor = torch.ones(1, patch_size_h, patch_size_w, patch_size_z).float().cuda()

    for j, feature_mask in enumerate(masks_valid):
        print_log('-------- input_modalities: {} --------'.format(mask_name_valid[j]))
        conf_mat = np.zeros((num_cls, num_cls), dtype=int)
        for i, data in enumerate(val_loader):
            # Load data
            start = time.time()
            inputs = data[0].float().cuda()
            B, _, H, W, Z = inputs.size()
            target = data[1].long().cuda()
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(B, 1)
            mask = mask.bool().cuda()
            # print('val inputs:{} target:{} '.format(inputs.shape, target.shape))
            gt = target[0].data.cpu().numpy().argmax(axis=0).astype(np.uint8)
            gt_idx = gt < num_cls  # Ignore every class index larger than the number of classes
            
            # Get sliding window indices
            h_idx_list = list(np.arange(0, H - patch_size_h + 1, int(patch_size_h * 0.5)))
            w_idx_list = list(np.arange(0, W - patch_size_w + 1, int(patch_size_w * 0.5)))
            z_idx_list = list(np.arange(0, Z - patch_size_z + 1, int(patch_size_z * 0.5)))

            h_idx_list.append(H - patch_size_h)
            w_idx_list.append(W - patch_size_w)
            z_idx_list.append(Z - patch_size_z)

            # Compute weight matrix
            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z] += one_tensor
            weight = weight1.repeat(B, num_cls, 1, 1, 1)

            # Sliding window inference
            pred = torch.zeros(B, num_cls, H, W, Z).float().cuda()
            segmenter.module.is_training = False
            
            with torch.no_grad():        
                for h in h_idx_list:
                    for w in w_idx_list:
                        for z in z_idx_list:
                            x_input = inputs[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z]
                            pred_part = segmenter(x_input, mask)
                            pred[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z] += pred_part
                            
            pred = pred / weight
            pred = pred[:, :, :H, :W, :Z]

            pred = pred[0].data.cpu().numpy().argmax(axis=0).astype(np.uint8)
            
            # Compute conf_mat
            conf_mat += confusion_matrix(gt[gt_idx], pred[gt_idx], num_cls)

            batch_time.update(time.time() - start)
        
        glob, mean, iou, dice = getScores(conf_mat)
        print_log('Epoch %-4d  glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f   Dice=%-5.2f' %
            (current_epoch+1, glob, mean, iou, dice))
        dice_modality[j] = dice
        val_dice_score.update(dice)

    for z, mask_name in enumerate(mask_name_valid):
        writer.add_scalar('{}_dice'.format(mask_name), dice_modality[z], global_step=current_epoch+1)
    writer.add_scalar('average_dice', val_dice_score.avg, global_step=current_epoch+1)
    print_log('Avg Dice scores: {}'.format(val_dice_score.avg))

    print_log('Val time per batch: {}'.format(batch_time.avg))
    print_log('Val time per epoch: {}'.format(time.time() - val_start))

    segmenter.train()
    return val_dice_score.avg


def evaluate(segmenter, dataset, test_loader, save_dir, patch_size, save_image=False, note='test'):
    if dataset in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        mask_name_test = ['t2', 't1c', 't1', 'flair',
                't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                'flairt1cet1t2']
    elif dataset in ['CC2024']:
        mask_name_test = ['dwi', 't2', 't2fs', 't1ce',
                    't2dwi', 't2t2fs', 't1cet2fs', 't2fsdwi', 't1cedwi', 't1cet2',
                    't1cet2t2fs', 't1cet2fsdwi', 't1cet2dwi', 't2t2fsdwi',
                    't1cet2t2fsdwi']
    masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
            [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
            [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
            [True, True, True, True]]
    
    test_loader.dataset.set_stage('test')
    segmenter.eval()
    csv_name = os.path.join(save_dir, 'results_{}.csv'.format(note))

    file = open(csv_name, "a+")
    csv_writer = csv.writer(file)
    if dataset in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        csv_writer.writerow(['Subject', 'WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95', 'ETPro HD95'])
    elif dataset in ['CC2024']:
        csv_writer.writerow(['Subject', 'Dice', 'HD95'])
    file.close()

    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    test_start = time.time()
    for i, mask in enumerate(masks_test[::-1]):
        print_log('-------- input_modalities: {} --------'.format(mask_name_test[::-1][i]))
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow([mask_name_test[::-1][i]])
        file.close()
        with torch.no_grad():
            if dataset in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
                dice_score, hd95_score = test_dice_hd95_softmax(
                                test_loader,
                                segmenter,
                                feature_mask = mask,
                                csv_name = csv_name,
                                patch_size = patch_size,
                                save_image = save_image,
                                mask_name = mask_name_test[::-1][i])
            elif dataset in ['CC2024']:
                dice_score, hd95_score = test_dice_hd95_softmax_cc(
                                test_loader,
                                segmenter,
                                feature_mask=mask,
                                csv_name=csv_name,
                                patch_size = patch_size,
                                save_image = save_image,
                                mask_name = mask_name_test[::-1][i]
                                )
                
        test_dice_score.update(dice_score)
        test_hd95_score.update(hd95_score)

    print_log('Avg Dice scores: {}'.format(test_dice_score.avg))
    print_log('Avg HD95 scores: {}'.format(test_hd95_score.avg))
    print_log('Test time : {}'.format(time.time() - test_start))


def main():
    # Set args
    global args, writer
    args = get_arguments()
    if args.dataset in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        args.num_classes = 4
    elif args.dataset in ['CC2024']:
        args.num_classes = 2
    else:
        print ('dataset is error')
        exit(0)

    # Set log
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.system('cp -r *py models utils data %s' % save_dir)
    helpers.logger = open(os.path.join(save_dir, 'log.txt'), 'a+')
    print_log(' '.join(sys.argv))
    print_log(str(args))
    writer = SummaryWriter(os.path.join(save_dir, 'log'))
    ckpt_dir = os.path.join(save_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Generate Segmenter
    torch.cuda.set_device(args.gpu[0])
    segmenter, param_groups = create_segmenter(args.model, args.crop_size[0], args.num_classes, args.gpu)
    if args.print_network:
        print_log(segmenter)
        # from thop import profile
        # input = torch.randn(1, 4, 80, 80, 80)
        # mask = torch.unsqueeze(torch.from_numpy(np.array([True, True, True, True])), dim=0)
        # flops, params = profile(segmenter, (input, mask))
        # print_log('Flops: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))
        # torch.cuda.synchronize()
        # start = time.time()
        # input.cuda()
        # mask.cuda
        # _ = segmenter(input, mask)
        # torch.cuda.synchronize()
        # end = time.time()
        # print_log('infer_time:', end-start)
        # exit(0)
    print_log('Loaded Segmenter {}, Pretrained={}, #PARAMS={:3.2f}M'
          .format(args.model, args.pretrained, compute_params(segmenter) / 1e6))
    
    # LR_Scheduler and Optimiser
    optimizer = PolyWarmupAdamW(
        # encoder,encoder-norm,decoder
        params=[
            {
                "params": param_groups,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.999],
        warmup_iter=args.iter*args.warmup,
        max_iter=args.iter*args.num_epochs,
        warmup_ratio=1e-6,
        power=0.9,
    )

    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(args.resume, {'segmenter': segmenter, 'optimizer': optimizer})
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return
    
    if args.pretrained:
        if os.path.isfile(args.pretrained_path):
            ckpt = torch.load(args.pretrained_path, map_location='cpu')

            if 'segmenter' in ckpt:
                state_dict = ckpt['segmenter']

                # # Handle cases where the state_dict might be wrapped in "module."
                # new_state_dict = {}
                # for param_name, param_value in state_dict.items():
                #     if param_name.startswith("module."):
                #         new_state_dict[param_name[7:]] = param_value
                #     else:
                #         new_state_dict[param_name] = param_value

                # # Load the parameters into the segmenter model
                # segmenter.load_state_dict(new_state_dict, strict=False)

                missing_keys, unexpected_keys = segmenter.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print_log(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print_log(f"Unexpected keys: {unexpected_keys}")

                print_log(f"=> Pretrained segmenter weights loaded from '{args.pretrained_path}'")
            else:
                print_log(f"=> No segmenter found in the pretrained checkpoint at '{args.pretrained_path}'")
        else:
            print_log(f"=> Pretrained path '{args.pretrained_path}' not found")


    # Saver
    saver = Saver(args=vars(args), ckpt_dir=ckpt_dir, best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score

    # Train and validation
    start = time.time()
    torch.cuda.empty_cache()
    # Create dataloaders
    train_loader, val_loader, test_loader = create_loaders(args.train_dir, args.val_dir, args.test_dir, 
                                                            args.train_list, args.val_list, args.test_list, 
                                                            args.crop_size, args.batch_size, args.num_workers, args.num_classes)
    if args.evaluate:
        print_log('############# Test #############')
        return evaluate(segmenter, args.dataset, test_loader, save_dir, args.crop_size, save_image=args.save_image, note='test')

    # Start training
    print_log('############# Training #############')
    train_iter = iter(train_loader)
    for epoch in range(epoch_start, args.num_epochs):
        train(segmenter, train_loader, train_iter, optimizer, args.num_epochs, epoch, args.iter, args.rfse, args.num_classes, args.alpha, args.print_loss)
        # validation
        if (args.validate == True) and (epoch + 1) % (args.val_every) == 0:
            print_log('############# Validation #############')
            dice = validate(segmenter, args.dataset, val_loader, epoch, args.num_classes, args.crop_size)
            saver.save(dice, {'segmenter' : segmenter.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch_start' : epoch+1})
        else:
            saver.save(0, {'segmenter' : segmenter.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch_start' : epoch+1})
        if (epoch + 1) % (args.save_every) == 0:
            torch.save({'best_val': 0, 'segmenter' : segmenter.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch_start' : epoch+1}, 
                       '{}/model-{}.pth.tar'.format(ckpt_dir, epoch+1))
        torch.cuda.empty_cache()
    print_log('Training stage finished, time spent {:.3f} hours, best val dice is {:.3f}\n'.format((time.time() - start) / 3600., saver.best_val))

    # Test the last epoch model
    print_log('############# Test Last Epoch #############')
    evaluate(segmenter, args.dataset, test_loader, save_dir, args.crop_size, save_image=args.save_image, note='last_epoch')
    
    # Test the best epoch model
    if (args.validate == True):
        print_log('############# Test Best Epoch #############')
        ckpt = os.path.join(ckpt_dir, 'model-best.pth.tar')
        if os.path.isfile(ckpt):
            _, _ = load_ckpt(ckpt, {'segmenter': segmenter})
            evaluate(segmenter, args.dataset, test_loader, save_dir, args.crop_size, save_image=args.save_image, note='best_epoch')
        else:
            print_log("=> no checkpoint found at '{}'".format(ckpt))
            return
    
    helpers.logger.close()


if __name__ == '__main__':
    main()