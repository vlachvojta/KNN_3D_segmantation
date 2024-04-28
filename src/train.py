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

import argparse
import numpy as np
import time

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from data_loader import DataLoader as CustomDataLoader
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    # TODO add args for dataset points per object, voxel size, click area
    parser.add_argument("-m", "--pretrained_model_path", required=True,
                        help="Pretrained model path (required)")
    parser.add_argument('-o', '--output_dir', type=str, default='../training/basic',
                        help='Where to store training progress.')
    parser.add_argument('-s', '--save_step', type=int, default=50,
                        help='How often to save checkpoint')
    parser.add_argument('-t', '--test_step', type=int, default=10,
                        help='How often to test model with validation set')

    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    return parser.parse_args()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO if --output_dir exists, LOAD model FROM CHECKPOINT
    # load model from path
    inseg_model_class, inseg_global_model = get_model(args.pretrained_model_path, 'cpu') #, device)

    optimizer = optim.SGD(
        inseg_global_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_dataloader = CustomDataLoader(args.dataset_path, points_per_object=5, verbose=False)
    val_dataloader = CustomDataLoader(args.dataset_path, points_per_object=5)  # TODO change dataset_path to validation set

    train_step = 0  # TODO get number of already trained steps if loading trained checkpoint
    val_ious, train_losses = [], []
    voxel_size = 0.05  # TODO get voxel size from args
    test_step_time = time.time()

    for epoch in range(args.max_epochs):
        train_dataloader.new_epoch()
        epoch_time = time.time()

        while True:  # get new batches until they run out
            train_batch = train_dataloader.get_batch(args.batch_size)
            if not train_batch: break

            if train_step % args.test_step == 0:
                print(f'\nEpoch: {epoch} train_step: {train_step}, Loss: {sum(train_losses[:args.test_step]) / args.test_step}'
                      f', time: {time.time() - test_step_time:.2f} seconds')
                val_iou = test_step(inseg_model_class, inseg_global_model, val_dataloader)
                val_ious.append(val_iou)
                plot_stats(train_losses, val_ious, train_step)
                test_step_time = time.time()

            if train_step % args.save_step == 0:
                save_step(inseg_global_model, args.output_dir, train_step)

            coords, feats, labels = train_batch
            sinput = create_input(feats, coords, voxel_size)
            labels = labels_to_logit_shape(labels)

            inseg_global_model.train()
            out = inseg_global_model(sinput).slice(sinput)

            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels)
            loss.backward()
            optimizer.step()
            print(f'loss: {loss.item():.2f}')
            train_losses.append(loss.item())
            train_step+=1
            print('.', end='', flush=True)

        print(f'\n\nEpoch {epoch} took {time.time() - epoch_time:.2f} seconds\n')

def get_model(pretrained_weights_file, device):
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    return inseg_global, global_model

def labels_to_logit_shape(labels: torch.Tensor):
    if len(labels.shape) == 3:
        labels = labels[0]

    labels_new = torch.zeros((len(labels), 2))
    labels_new[labels[:, 0] == 0, 0] = 1
    labels_new[labels[:, 0] == 1, 1] = 1
    return labels_new

def create_input(feats, coords, voxel_size: int = 0.05):
    if len(feats.shape) == 3:
        feats = feats[0]
    if len(coords.shape) == 3:
        coords = coords[0]

    sinput = ME.SparseTensor(
        features=feats,
        coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        # device=device
    )  # .to(device)
    return sinput

def test_step(model_class, model, val_dataloader):
    model.eval()

    # TODO do forward pass for every data from eval set, get accuracy statistic
    ious = []
    # while True:
    #     batch = val_dataloader.get_batch(args.batch_size)
    #     if not train_batch: break

    #     coords, feats, labels = batch
    #     coords = coords[0]
    #     ...

    #     pred = model_class.prediction(feats, coords, model)
    #     iou = model_class.mean_iou(pred, labels)
    #     ious.append(iou)

    # TODO export few examples of rendered result images using utils.save_point_cloud_views

    return sum(ious) / len(ious) if len(ious) > 0 else 0

def save_step(inseg_global_model, path, train_step):
    ...
    # TODO save model checkpoint to os.path.join(path, f'model_{train_step}.pth')

def plot_stats(train_losses, val_ious, train_step):
    print(f'\nTest step. Train losses: {train_losses}') #, Val IoUs: {val_ious}')
    # TODO produce chart of train_losses and val_ious
    # TODO also save train_losses and val_ious to .npy or something for future reference

if __name__ == '__main__':
    main(parse_args())
