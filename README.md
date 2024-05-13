# KNN_3D_segmentation: Interaktivní segmentace bodových mračen
*KNN project (Convolutional Neural Networks) at FIT (B|V)UT. 2023/2024 summer semestr.*

## Autoři:
- Zuzana Hrkľová xhrklo00
- Martin Kneslík xknesl02
- Vojtěch Vlach xvlach22

## Zadání

Libovolnou segmentační úlohu lze změnit na interaktivní tím, že na vstup sítě nedám jen obraz, ale i uživatelský vstup, třeba jako další "barevný" kanál s místy, které uživatel označil. Podobně to jde u bodových mrače. Můžete využít existující datasety (např. [KITTI](http://www.cvlibs.net/datasets/kitti/eval_semantics.php), [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [NYU Depth V2 - Kaggle](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)), nebo si i můžete pujčit LIDAR Livox Horizon, přípdaně nějakou RGB-D kameru typu Kinect.
Ning Xu, Brian Price, Scott Cohen, Jimei Yang, and Thomas Huang. Deep Interactive Object Selection. CVPR 2016. https://sites.google.com/view/deepselection

## Dataset S3DIS
- [paperswithcode.com/dataset/s3dis](https://paperswithcode.com/dataset/s3dis)
- point cloud dataset
- 6 vnitřních komplexůs 271 místnostmi
- prostor je rozdělen do 13 sémantických kategorií

## Architektura modelu
Použili jsme předtrénovaný MinkUnet (Minkowski Engine), což je síť na styl Unetu, ale využívající sparse konvoluce a sparse tensory vhodné pro řídce uložená data.

## Zdroje
Článek, ze kterého vycházíme: [Interactive Object Segmentation in 3D Point Clouds, článek](https://arxiv.org/pdf/2204.07183.pdf)

Minkowski engine (torch extension): [docu](https://nvidia.github.io/MinkowskiEngine/) [GitHub](https://github.com/NVIDIA/MinkowskiEngine)

Kód ve složce `src/InterObject3D` byl převzat a následně upraven z repozitáře [github.com/theodorakontogianni/InterObject3D](https://github.com/theodorakontogianni/InterObject3D)
