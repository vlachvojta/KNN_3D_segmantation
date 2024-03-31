import argparse
from data_loader import DataLoader

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Refresh DataLoader cache")
    args = parser.parse_args()

    data_loader = DataLoader(args.src_path, 5, args.force)

    coords_batch, feats_batch, label_batch = data_loader.get_batch(5)

    print(f'coords_batch ({type(coords_batch)}): {coords_batch.shape}')
    print(f'feats_batch ({type(feats_batch)}): {feats_batch.shape}')
    print(f'label_batch ({type(label_batch)}): {label_batch.shape}')

    # Example usage, get every batch
    # while True:
    #     batch = data_loader.get_batch(5)
    #     if not batch:
    #         break
    #     print(batch)
