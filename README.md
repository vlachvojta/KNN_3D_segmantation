# KNN_3D_segmantation
KNN project (Convolutional Neural Networks) at FIT (B|V)UT. 2023/2024 summer semestr.

## Zadání

Libovolnou segmentační úlohu lze změnit na interaktivní tím, že na vstup sítě nedám jen obraz, ale i uživatelský vstup, třeba jako další "barevný" kanál s místy, které uživatel označil. Podobně to jde u bodových mrače. Můžete využít existující datasety (např. [KITTI](http://www.cvlibs.net/datasets/kitti/eval_semantics.php), NYU Depth Dataset V2 [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [NYU Depth V2 - Kaggle](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)), nebo si i můžete pujčit LIDAR Livox Horizon, přípdaně nějakou RGB-D kameru typu Kinect.
Ning Xu, Brian Price, Scott Cohen, Jimei Yang, and Thomas Huang. Deep Interactive Object Selection. CVPR 2016. https://sites.google.com/view/deepselection


## TODOs
- [ ] vybrat dataset
- [ ] vybrat prostředí experimentu a architekturu
- [ ] vytvořit neinteraktivní baseline


## Datasety

- NUY - RGBD
- KITTI - point cloudy
- S3DIS - https://paperswithcode.com/dataset/s3dis
- Předzpracování / augmentace: zkusit různý rotace, skew, shift… (viz PointNet článek)
- Augmentace: jak zadávat náhodně negativní vstupy okolo…
- Na vstup dát i normály (z RGB-D do point cloudu to musíme dopočítat/doodhadnout třeba pomocí open3D)


## Rámcový postup

1) Baseline:
    - Neinteraktivní segmentace na point cloudech
2) Interactive rozšíření
    - Trénování pomocí samplování jedné třídy z původního datasetu
    - Jenom pozitivní body
3) Rozšíření o negativní body
    - Přidat negativní body do trénovací sady
4) Přidat možnost víc bodů
    - Input je už částečně obarvenej point cloud + nový bod pozitivní / negativní

## Architektura modelu
- Grafové neuronové sítě
- Existují i speciální sítě na pointCloudy viz PointNet, PointNet++
- Popř. Transformery

## Prostředí experimentů
- Open3D-ML
- PyTorch
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) - support přímo na point cloudy…

