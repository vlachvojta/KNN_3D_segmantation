# Dataset
1. Download [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset 
2. Unzip the dataset here (dataset/Stanford3dDataset_v1.2/)
3. Run [src/convert_dataset.py](../src/convert_dataset.py) script. 
    * The script computes normals, converts files to better format (.pcd) and creates downsampled annotations for non-interactive baseline
