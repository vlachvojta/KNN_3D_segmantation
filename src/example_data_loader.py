import argparse
from data_loader import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Refresh DataLoader cache")
    args = parser.parse_args()

    data_loader = DataLoader(args.src_path, 5, args.force)

    batch = data_loader.get_batch(5)

    print("batch type: ", type(batch))
    print("batch label type: ", type(batch[0][1]))
    print("batch input type: ", type(batch[0][0]))

    for input, label in batch:
        print(input.point)
        break

    # Example usage, get every batch
    # while True:
    #     batch = data_loader.get_batch(5)
    #     if not batch:
    #         break
    #     print(batch)
