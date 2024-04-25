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

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from data_loader import DataLoader as CustomDataLoader
import utils

# from examples.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    # parser.add_argument("-m", "--checkpoint_dir", default="../checkpoints",
    #                     help="Model path (required)")
    parser.add_argument("-m", "--pretrained_model_path", required=True,
                        help="Pretrained model path (required)")
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    return parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model from path
    inseg_model_class, inseg_global_model = get_model(args.pretrained_model_path, 'cpu') #, device)

    optimizer = optim.SGD(
        inseg_global_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_dataloader = CustomDataLoader(args.src_path, points_per_object=5)

    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(args.max_epochs):
        while True:
            train_batch = train_dataloader.get_batch(1)
            if not train_batch:
                break

            coords, feats, labels = train_batch
            coords = coords[0]
            feats = feats[0]
            labels = labels[0]
            labels = labels.float().long()
            print(f'Batch: {coords.shape=}, {feats.shape=}, {labels.shape=}')

            voxel_size = 0.05
            # Feed-forward pass and get the prediction
            # sinput = ME.SparseTensor(feats, coords)  # jednodussi varianta, ale taky to padalo na nejakej error
            sinput = ME.SparseTensor(
                features=feats,
                coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                # device=device
            )  # .to(device)


            print(f'sinput.F.shape: {sinput.F.shape}')
            inseg_global_model.train()
            out = inseg_global_model(sinput)
            print('out.F.shape:', out.F.shape)
            out = out.slice(sinput)
            print('out.F.shape after slice:', out.F.shape)
            optimizer.zero_grad()
            print('out.F.squeeze().shape:', out.F.squeeze().shape)
            print('labels.long().shape:', labels.long().shape)
            loss = criterion(out.F.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
                )
                accum_loss, accum_iter = 0, 0

def get_model(pretrained_weights_file, device):
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    return inseg_global, global_model


if __name__ == '__main__':
    main(parse_args())
