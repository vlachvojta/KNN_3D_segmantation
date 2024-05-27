# This file is a modified version of the training script from the MinkowskiEngine repository.
# The original script is available at https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/training.py

# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import os
import sys
import argparse
import time
import re

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import open3d as o3d

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from InterObject3D import minkunet
from data_loader import DataLoader as CustomDataLoader
import compute_iou
import utils


def parse_args():
    print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", default="../dataset/S3DIS_converted_separated/train",
                        help="Source dataset path. (default: ../dataset/S3DIS_converted_separated/train)")
    parser.add_argument("-vd", "--val_dataset", default="../dataset/S3DIS_converted_separated/validation",
                        help="Validation dataset path (default: ../dataset/S3DIS_converted_separated/validation")
    parser.add_argument("-v", "--voxel_size", default=0.05, type=float,
                        help="The size data points are converting to (default: 0.05)")
    parser.add_argument("-c", "--click_area", default=0.3, type=float,
                        help="Area of the simulated click points. MUST BE LARGER THAN VOXEL SIZE (default: 0.3)")

    parser.add_argument("-m", "--pretrained_model_path", type=str, default=None,
                        help="Pretrained model path to start training with (default: None)")
    parser.add_argument("-mc", "--model_class", type=str, default='MinkUNet34C',
                        help="Model class to use (default: MinkUNet34C)")
    parser.add_argument('-o', '--output_dir', type=str, default='../training/InterObject3D_test',
                        help='Where to store training progress.')
    parser.add_argument('-l', '--validation_out', type=str, default='../val_results/',
                        help='Where to store validation results.')
    parser.add_argument('-s', '--save_step', type=int, default=50,
                        help='How often to save checkpoint')
    parser.add_argument('-t', '--test_step', type=int, default=10,
                        help='How often to test model with validation set')
    parser.add_argument('-g', '--stats_path', type=str, default='../stats',
                        help='Where to store training stats')
    parser.add_argument('-sl', '--saved_loss', type=str, default=None,
                        help='Path to saved training loss data from previous training')
    parser.add_argument('-siv', '--saved_ious_val', type=str, default=None,
                        help='Path to saved IOU data from previous training')
    parser.add_argument('-sit', '--saved_ious_train', type=str, default=None,
                        help='Path to saved IOU data from previous training')
    parser.add_argument('-b', '--batch_size', default=20, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()
    print(f'args: {args}')
    return args

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Using device: {device}')
    utils.ensure_folder_exists(args.output_dir)
    utils.ensure_folder_exists(args.stats_path)

    inseg_model_class, inseg_global_model, train_step = get_model(args.pretrained_model_path, args.output_dir, args.model_class, device)

    optimizer = optim.SGD(
        inseg_global_model.parameters(),
        lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_dataset = CustomDataLoader(args.dataset_path, verbose=False, click_area=args.click_area, normalize_colors=True, voxel_size=args.voxel_size)

    # create cache for validation dataset
    val_dataloader = CustomDataLoader(args.val_dataset, verbose=False, click_area=args.click_area, limit_to_one_object=True, normalize_colors=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=ME.utils.batch_sparse_collate)

    train_losses, val_ious, train_ious = load_stats(args.saved_loss, args.saved_ious_val, args.saved_ious_train)
    train_ious_before_slice = []
    voxel_size = args.voxel_size
    test_step_time = time.time()
    start_time = time.time()

    train_steps_in_epoch = train_dataset.remaining_unique_elements() // args.batch_size
    print(f'Train steps in one epoch: {train_steps_in_epoch}')
    print(f'Training started at {time.ctime()}\n')

    for epoch in range(args.max_epochs):
        train_dataset.new_epoch()
        epoch_time = time.time()
        train_iter = iter(train_dataloader)
        inseg_global_model.train()

        for _ in range(train_steps_in_epoch):
            train_batch = next(train_iter)

            if train_step % 5 == 0:
                saved = empty_cache()
                print(f'Saved {saved} GB of memory by emptying cache.')
                # torch.cuda.empty_cache()  # release unassigned variables/tensors from GPU memory

            if args.test_step > 0 and train_step % args.test_step == 0:
                inseg_global_model.eval()
                print('\n\n-------------------------------------------------------------------------------------')
                print(f'Epoch: {epoch} train_step: {train_step}, mean loss: {sum(train_losses[-args.test_step:]) / args.test_step:.2f}, '
                      f'time of test_step: {utils.timeit(test_step_time)}, '
                      f'time from start: {utils.timeit(start_time)}')
                # val_iou = test_step(inseg_model_class, inseg_global_model, val_dataloader)
                iou_args = {'src_path': args.val_dataset, 
                            'model_path': " ", 
                            'output_dir': f'{args.validation_out}_{train_step}', 
                            'inseg_model': inseg_model_class, 
                            'inseg_global': inseg_global_model,
                            'show_3d': False,
                            'limit_to_one_object': True,
                            'verbose': False,
                            'max_imgs': 3,
                            'click_area': args.click_area,
                            'voxel_size': voxel_size}
                val_iou = compute_iou.main(iou_args)
                val_ious.append(val_iou)
                print(f'Validation finished with mean IOU: {val_iou}')
                plot_stats(train_losses, val_ious, train_ious, train_step, args.stats_path)
                test_step_time = time.time()
                print('-------------------------------------------------------------------------------------\n')
                inseg_global_model.train()

            if train_step % args.save_step == 0:
                save_step(inseg_global_model, args.output_dir, train_step)

            train_step+=1

            # point cloud inputs
            coords, feats, labels = train_batch
            labels = labels_to_logit_shape(labels)
            labels = labels.float()
            feats = feats.float()
            print(f'{coords.shape=}, {feats.shape=}, {labels.shape=}')

            print(f'Before adding data: memory_allocated: {mem_aloc()}, memory_cached(): {mem_cache()}')
            if mem_cache() > 7:
                print('!!! Clearing cache !!!')
                # torch.cuda.empty_cache()
                saved = empty_cache()
                print(f'Saved {saved} GB of memory by emptying cache.')
                print(f'After clearing cache memory_allocated: {mem_aloc()}, memory_cached(): {mem_cache()}')
            # voxelized input
            super_feats = torch.cat((feats, labels), dim=1)
            super_sinput = ME.SparseTensor(super_feats.float(), coords, device=device)
            sinput = ME.SparseTensor(super_sinput.F[:, :-2], super_sinput.C, device=device)
            slabels = ME.SparseTensor(super_sinput.F[:, -2:], super_sinput.C, device=device)
            print(F'{sinput.F.shape=}, {slabels.F.shape=}')
            if not clicks_in_sinput(sinput, args.batch_size) or not labels_in_sinput(slabels) or tensor_too_big(sinput, 610000):
                continue

            # voxelized output
            sout = inseg_global_model(sinput)
            optimizer.zero_grad()
            # sout_for_loss = torch.softmax(sout.F, dim=1)
            # loss = criterion(sout_for_loss, slabels.F)
            loss = criterion(sout.F, slabels.F)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_iou_before_slice = inseg_model_class.mean_iou(sout.F.argmax(dim=1), slabels.F.argmax(dim=1)).cpu()
            print(f'After adding data: memory_allocated: {mem_aloc()} GB, memory_cached(): {mem_cache()} GB')
            # save first 10 voxelized point clouds (first out of every batch) for every test_step
            if args.test_step > 0 and train_step % args.test_step < 5:
                train_step_to_save = train_step - (train_step %  args.test_step)
                visualize_one_voxelized_point_cloud(sinput, slabels, sout, train_iou_before_slice,
                                                    os.path.join(args.output_dir, f'train_results_{train_step_to_save}'),
                                                    train_step %  args.test_step)

            # point cloud output
            out = sout.slice(super_sinput)
            out = out.F.argmax(dim=1).cpu()
            labels = labels.argmax(dim=1)
            train_iou = inseg_model_class.mean_iou(out, labels).cpu()
            train_ious.append(train_iou)
            print(f'train_loss: {loss.item():.5f}, train_iou_before_slice: {train_iou_before_slice:.5f}, train_iou: {train_iou:.5f}')
            print('.', end='', flush=True)
            print('')

        print(f'\n\nEpoch {epoch} took {utils.timeit(epoch_time)}')

def empty_cache():
    before = mem_cache()
    torch.cuda.empty_cache()
    after = mem_cache()
    return round(before - after, 2)

def bytes_to_gb(bytes):
    return round(bytes / 1024 / 1024 / 1024, 2)

def mem_aloc():
    return bytes_to_gb(torch.cuda.memory_allocated())

def mem_cache():
    return bytes_to_gb(torch.cuda.memory_cached())

def get_model(pretrained_weights_file, output_dir, model_class, device):
    # try to find model in output_dir
    model_regex = r'(model|MinkUNet\d{2,3}[A-Z]?)_(\d+).pth'
    models = [re.match(model_regex, f) for f in os.listdir(output_dir) if re.match(model_regex, f)]

    trained_steps = 0
    if models:
        model_dict = {int(model.group(2)): model.group(0) for model in models}
        trained_steps = max(model_dict.keys())
        pretrained_weights_file = os.path.join(output_dir, model_dict[trained_steps])

    print(f'Loading model from {pretrained_weights_file}')
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(inseg_global.pretraining_weights_file, model_class, device)
    return inseg_global, global_model, trained_steps

def load_stats(saved_loss, val_ious, train_ious):
    if (saved_loss == None or val_ious == None or train_ious == None):
        return [], [], []
    else:
        losses = load_from_numpy_ignore_missing(saved_loss, default=[])
        val_ious = load_from_numpy_ignore_missing(val_ious, default=[])
        train_ious = load_from_numpy_ignore_missing(train_ious, default=[])
        return losses, val_ious, train_ious

def load_from_numpy_ignore_missing(path, default=None):
    if os.path.exists(path):
        data = np.load(path)
        if default == []:
            return list(data)
        return data
    return default

def labels_to_logit_shape(labels: torch.Tensor):
    if len(labels.shape) == 3:
        return tuple(labels_to_logit_shape(label) for label in labels)

    labels_new = torch.zeros((len(labels), 2))
    labels_new[labels[:, 0] == 0, 0] = 1
    labels_new[labels[:, 0] == 1, 1] = 1
    return labels_new

def clicks_in_sinput(sinput, batch_size) -> bool:
    assert sinput.F.shape[1] == 5, f'Expected 5 features in sinput (RGB, P+N clicks), got {sinput.F.shape[1]}'

    positive_click_count = torch.sum(sinput.F[:, 3] != 0)
    # negative_click_count = torch.sum(sinput.F[:, 4] != 0)
    if positive_click_count == 0:  # TODO Check with batch_size somehow
        print(f'!!! Skipping batch !!! Not enough positive clicks in sinput.')
        return False
    # elif positive_click_count < 4:
    #     print(f'Not many positive clicks found in sinput (voxelized point cloud) : (positive: {positive_click_count}, negative {negative_click_count}).')

    return True

def labels_in_sinput(slabels) -> bool:
    local_labels = slabels.F.argmax(dim=1)
    non_zero_labels = torch.sum(local_labels != 0)
    numel = local_labels.numel()
    if non_zero_labels / numel < 0.004:  # less than 0.4% of labels are non-zero
        # print(f'from {local_labels.shape} labels, {non_zero_labels} are non-zero ({non_zero_labels / numel * 100:.2f}%)')
        print(f'!!! No labels in slabels, skipping!!! ({non_zero_labels / numel * 100:.2f}%)')
        return False

    return True

def tensor_too_big(sinput, max_size) -> bool:
    if sinput.F.shape[0] > max_size:
        print(f'!!! Skipping batch !!! More elements then max_size: {sinput.F.shape[0]} > {max_size}')
        return True
    return False


def save_step(model, path, train_step):
    export_path = os.path.join(path, f'{model.__class__.__name__}_{train_step}.pth')
    torch.save(model.state_dict(), export_path)
    print(f'Model saved to: {export_path}\n')

def plot_stats(train_losses, val_ious, train_ious, train_step, graphs_path):
    # train_losses_str = ', '.join([f'{loss:.5f}' for loss in train_losses])
    # print(f'\nTest step. Train losses: [{train_losses_str}]') # , Val IoUs: {val_ious}')

    # print(f'train_losses: {train_losses}')
    # print(f'val_ious: {val_ious}')

    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    plot_stats_values(train_losses, ax[0, :], title='Train losses', xlabel='Training step')
    plot_stats_values(train_ious, ax[1, :], title='Train IOU', xlabel='Training step')
    plot_stats_values(val_ious, ax[2, :], title='Validation IOU', xlabel='Test step')

    plt.tight_layout()
    print(f'Saving losses to {os.path.join(graphs_path, "losses.png")}')
    plt.savefig(os.path.join(graphs_path, 'losses.png'))
    plt.clf()

    np.save(os.path.join(graphs_path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(graphs_path, 'val_ious.npy'), val_ious)
    np.save(os.path.join(graphs_path, 'train_ious.npy'), train_ious)

def running_average(stats, window_size_ratio=0.1):
    window_size = int(len(stats) * window_size_ratio)
    running_avg = np.zeros(len(stats))
    running_avg[0] = stats[0]

    for i in range(1, len(stats)):
        if i < window_size:
            running_avg[i] = np.mean(stats[:i])
        else:
            running_avg[i] = np.mean(stats[i-window_size:i])

    return running_avg

def plot_stats_values(stats, ax, title='', xlabel=''):
    running_average_precalced = running_average(stats)
    for i in range(2):
        ax[i].plot(stats, label='IoU', color='b')
        ax[i].axhline(y=np.max(stats), color='g', linestyle='--')
        ax[i].axhline(y=np.min(stats), color='g', linestyle='--')
        ax[i].plot(stats, color='b')
        ax[i].plot(running_average_precalced, color='r', linewidth=2)
        ax[i].set_title(title)
        ax[i].set_xlabel(xlabel)
        ax[i].set_title(title)
        ax[i].set_xlabel(xlabel)

    ax[1].set_yscale('log')

def visualize_one_voxelized_point_cloud(sinput, slabels, sout, iou, output_dir, i, show_3d=False, verbose=False):
    # select coords for first point cloud
    pcd_0_idx = sinput.C[:, 0] == 0
    coords = sinput.C[pcd_0_idx, 1:]
    feats = sinput.F[pcd_0_idx, :]
    labels = slabels.F[pcd_0_idx, :].argmax(dim=1)
    out = sout.F[pcd_0_idx, :].argmax(dim=1)
    # print(f'{coords.shape=}, {feats.shape=}, {labels.shape=}, {out.shape=}')

    pcd = utils.get_output_point_cloud(coords, feats, labels, out)
    utils.save_point_cloud_views(pcd, iou, i, output_dir, verbose)
    if show_3d:
        o3d.visualization.draw_geometries([pcd])
    # print(f'exit'); exit()

# def collation_fn(data_labels):
#     print(f'collation_fn:')
#     print(f'\tdata_labels: {type(data_labels)}')
#     print(f'\tdata_labels: {len(data_labels)}')
#     print(f'\tdata_labels[0][0]: {type(data_labels[0][0])}')
#     print(f'\tdata_labels[0][0].shape: {data_labels[0][0].shape}')
#     print(f'\tdata_labels[0][0].dtype: {data_labels[0][0].dtype}')

#     print(f'\tcoords of first element: {data_labels[0][0].shape=}, {data_labels[0][0].dtype=}')
#     print(f'\tfeats of first element: {data_labels[0][1].shape=}, {data_labels[0][1].dtype=}')
#     print(f'\tlabels of first element: {data_labels[0][2].shape=}, {data_labels[0][2].dtype=}')


#     # import IPython; IPython.embed()

#     # for dato in data_labels:
#     #     print(f'\t\tdato: {type(dato)}')
#     #     print(f'\t\tdato: {len(dato)}')
#     #     print(f'\t\tdato: {dato[0].shape=}, {dato[1].shape=}, {dato[2].shape=}')
#     coords, feats, labels = list(zip(*data_labels))
#     # print(f'coords: ({type(coords)}) {len(coords)}, feats: ({type(feats)}) {len(feats)}, labels: ({type(labels)}) {len(labels)}')
#     # print(f'coords[0]: ({type(coords[0])}) {coords[0].shape}, feats[0]: ({type(feats[0])}) {feats[0].shape}, labels[0]: ({type(labels[0])}) {labels[0].shape}')
#     # print(f'\tcoords: {coords.shape}')
#     # print(f'\tfeats: {feats.shape}')
#     # print(f'\tlabels: {labels.shape}')
#     coords_batch, feats_batch, labels_batch = [], [], []

#     # Generate batched coordinates
#     coords_batch = ME.utils.batched_coordinates(coords, dtype=torch.float32)

#     # print(f'coords_batch ({coords_batch.shape=}, {coords_batch.dtype=})')

#     # Concatenate all lists
#     feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
#     labels_batch = torch.from_numpy(np.concatenate(labels, 0))

#     print('\n\n Result of collation_fn:')
#     print(f'coords_batch ({coords_batch.shape=}, {coords_batch.dtype}):\n{coords_batch}')
#     print(f'feats_batch ({feats_batch.shape=}, {feats_batch.dtype}):\n{feats_batch}')
#     print(f'labels_batch ({labels_batch.shape=}, {labels_batch.dtype}):\n{labels_batch}')

#     # print(f'exit'); exit()

#     return coords_batch, feats_batch, labels_batch

# def compare_tensors(t1, t2, t1_name, t2_name):
#     print(f'\n\nComparing {t1_name} and {t2_name}')
#     print(f'{t1_name}.shape={t1.shape}, {t2_name}.shape={t2.shape}')
#     print(f'{t1_name}:\n{t1}')
#     print(f'{t2_name}:\n{t2}')
#     print(f'!!! tensors equal?: {torch.all(t1 == t2)}')
#     sum_same = torch.sum(t1 == t2)
#     numel = t1.numel()
#     print(f'!!! sum of same values: {sum_same} out of {numel} ({sum_same / numel * 100:.2f}%)')
#     # label_loss = criterion(t1, t2)
#     # print(f'{label_loss=}')
#     # sum_ones_zero = torch.sum(t1[:, 0] == 1)
#     # sum_ones_one = torch.sum(t1[:, 1] == 1)
#     # valid_label = torch.sum(t1[:, 0] + t1[:, 1] == 1)
#     # print(f'{t1_name} ones_zero: {sum_ones_zero}, ones_one: {sum_ones_one} out of {numel}')
#     # print(f'{t1_name} valid labels: {valid_label} out of {len(t1)} ({valid_label / len(t1) * 100:.2f}%)')
#     non_zero = torch.sum(t1 != 0)
#     print(f'{t1_name} non_zero: {non_zero} out of {numel} ({non_zero / numel * 100:.2f}%)')
#     non_zero = torch.sum(t2 != 0)
#     print(f'{t2_name} non_zero: {non_zero} out of {numel} ({non_zero / numel * 100:.2f}%)')
#     print(f'Counting IOUfor {len(t1)} elements:')
#     intersection = torch.logical_and(t2, t1)
#     print(f'\nintersection: {torch.sum(intersection)}')
#     truepositive = intersection.sum()
#     union = torch.logical_or(t2, t1)
#     print(f'\nunion: {torch.sum(union)}')
#     union = torch.sum(union)
#     iou = 100 * (truepositive / union)
#     print(f'\nIOU: {iou:.2f}%')

# When theres too low IOUs...
# if train_iou < 0.000001 or train_iou_before_slice < 0.000001:

#     print(f'IOU is too low: analyzing tensors')
#     compare_tensors(out, labels, 'out', 'labels')
#     print(f'train_iou: {train_iou:.5f}, ')
#     compare_tensors(sout.F.argmax(dim=1),   slabels.F.argmax(dim=1),
#                    'sout.F.argmax(dim=1)', 'slabels.F.argmax(dim=1)')
#     print(f'train_iou_before_slice: {train_iou_before_slice:.5f}')

#     print('exiting'); exit()
    
#     utils.save_tensor_to_txt(out, 'out_2.txt')
#     utils.save_tensor_to_txt(labels, 'labels.txt')
#     concat = torch.cat((out.unsqueeze(1), labels.unsqueeze(1)), dim=1)
#     utils.save_tensor_to_txt(concat, 'concat.txt')
#     print(f'exiting'); exit()
if __name__ == '__main__':
    main(parse_args())
