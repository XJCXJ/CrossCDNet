_base_ = [
    '../_base_/models/resnet_ibn.py', '../common/standard_256x256_40k_levircd.py'
]
crop_size = (256, 256)
data_root = 'data/WHU-CD'
# neck = dict(
    # backbone=dict(
    # interaction_cfg=(
    #     None,
    #     dict(type='SpatialExchange', p=1/2),
    #     dict(type='ChannelExchange', p=1/2),
    #     dict(type='ChannelExchange', p=1/2)),
    # ),
    # decode_head=dict(
    #     num_classes=2,
    #     sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    # )
decode_head = dict(
    type='FPNHead',
    # priori_attn=True,
    # in_channels=[48, 96, 192, 1024],  # ShuffleNetV2
    in_channels=[128, 256, 512, 1024],  # resnet18
    # in_channels=[64, 128, 256, 512],  # resnet34
    # in_channels=[128, 256, 512, 1024],  # resnet34 - concat
    # in_channels=[512, 1024, 2048, 4096],  # resnet50
    # in_channels=[512, 1024, 2048, 4096],  # resnet101-concat
    # in_channels=[256, 512, 1024, 2048],  # resnet101-diff
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],  # 256/h/w
    channels=128,  # 48, 64,512,256
    dropout_ratio=0.1,
    num_classes=2,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=(256, 256)),
    dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(train=dict(pipeline=train_pipeline))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        img_dir='train',
        ann_dir='train/label',
        pipeline=train_pipeline),
    val=dict(
        img_dir='val',
        ann_dir='val/label',
        pipeline=test_pipeline),
    test=dict(
        img_dir='test_WHU',
        ann_dir='test_WHU/label',
        pipeline=test_pipeline))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.003,
    betas=(0.9, 0.999),
    weight_decay=0.05)

