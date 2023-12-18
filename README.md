# CrossCDNet

The Pytorch implementation for: "A Cross-domain Change Detection Network Based on Instance Normalization"

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
# refenrence
This project is implemented based on Open-CD
https://github.com/likyoo/open-cd
