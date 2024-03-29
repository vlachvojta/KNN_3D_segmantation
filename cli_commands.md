# CLI commands

- tady budou nějaký commandy, co budeme často používat...

## Instalace MinkowskiEngine s GPU
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

`sudo apt install --reinstall nvidia-driver-550`

`python [setup.py](http://setup.py/) install --blas=openblas`


## Instalace MinkowskiEngine bez GPU

```bash
sudo apt install build-essential python3-dev libopenblas-dev
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --cpu_only --blas=openblas
```
Pokud to hodí error "can't access directory {dir}", tak stačí změnint vlastníka složky pomocí: `sudo chown {user} {dir}`


## Příklad spuštění InterObject3D
```bash
cd InterObject3D/InterObject3D/Minkowski/training
èxport MODELS_3D={path_to_models}
python run_inter3d.py --verbal=True --instance_counter_id=1 --number_of_instances=1 --cubeedge=0.05 --pretraining_weights=$MODELS_3D/InterObject3D_pretrained/weights_exp14_14.pth --dataset='scannet'  --visual=True --save_results_file=True --results_file_name=results_scannet_mini.txt
```
