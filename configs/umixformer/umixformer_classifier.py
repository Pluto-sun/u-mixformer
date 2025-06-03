_base_ = [
    '../_base_/models/umixformer_classifier.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='UMixFormerClassifier',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mit_b0.pth')),
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1)

# data settings
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='ImageNet',
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        pipeline=test_pipeline),
    test=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        pipeline=test_pipeline))

# optimizer settings
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='accuracy') 