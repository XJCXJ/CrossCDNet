# CrossCDNet

The Pytorch implementation for: "A Cross-domain Change Detection Network Based on Instance Normalization"，
[pdf](https://www.mdpi.com/2072-4292/15/24/5785).

<div align="center">
  <img src="https://github.com/XJCXJ/CrossCDNet/blob/main/data/Fig.png">
</div>



# Usage
Train
```
python tools/train.py configs/resnet_ibn/train_.py --work-dir ./CrossCDNet_r18_levir_workdir
```

Test
```
python tools/test.py configs/resnet_ibn/train_.py  CrossCDNet_r18_levir_workdir/best_mIoU_iter_40000.pth'''
```
Visualization
```
python tools/test.py configs/resnet_ibn/train_.py  CrossCDNet_r18_levir_workdir/best_mIoU_iter_40000.pth --show-dir your_save_path
```
# Result
Train set : LEVIR-CD
<div align="center">
  <img src="https://github.com/XJCXJ/CrossCDNet/blob/main/data/Fig2.png">
</div>


# Refenrence
This project is implemented based on Open-CD
https://github.com/likyoo/open-cd

# Citation
If you find this project useful in your research, please consider cite:
```
@Article{rs15245785,
AUTHOR = {Song, Yabin and Xiang, Jun and Jiang, Jiawei and Yan, Enping and Wei, Wei and Mo, Dengkui},
TITLE = {A Cross-Domain Change Detection Network Based on Instance Normalization},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {24},
ARTICLE-NUMBER = {5785},
URL = {https://www.mdpi.com/2072-4292/15/24/5785},
ISSN = {2072-4292},
ABSTRACT = {},
DOI = {10.3390/rs15245785}
}
```
