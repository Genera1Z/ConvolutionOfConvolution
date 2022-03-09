### mobilenet_v2/mobilenet_v2_b32x8_imagenet.py

#
### _base_/models/shufflenet_v2_1x.py

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileNetV3Coc',  # XXX
        conv_cfg=dict(  # XXX
            type='Coc2d',
            sa=2, sk_plus=2,
            skip=True, dilat=True, wsc=False,
            share='ci',
            ws='LN', both=False,
            sbn='LN'
        ),
        arch='small'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=576,  # XXX
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
callbacks = [
    dict(type='DelCkptsHook', delta=30)
]

#
### _base_/datasets/imagenet_bs64_pil_resize.py

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,  # XXX b64x16 -> b128x8
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

#
### _base_/schedules/imagenet_bs1024_linearlr_bn_nowd.py

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    weight_decay=0.00004,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000,
)
runner = dict(type='EpochBasedRunner', max_epochs=300)

#
### _base_/default_runtime.py

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# You can register your own hooks like this
# custom_hooks=[dict(type='EMAHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
