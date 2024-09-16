name = 'mvimgnet_uni3d6-scratch'
work_dir = 'work_dirs/' + name

using_channel = 14

train_cfg = dict(
    epoch=50,
)

test_cfg = dict(
    test_gt=False,
    val_interval=5,# 10

    dataset="mvimgnet",
    task="retrive",
    test_interval=5,
)

model_cfg = dict(
    clip_model_path="/path/to/your/unigs/cache/ViT-B-16.pt",
    pointnet_model="uni3D",
    model_setting=dict(
        pc_model="eva02_small_patch14_224",
        pretrained_pc=None,
        drop_path_rate=0.2,

        pc_feat_dim=384,
        embed_dim=512,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        patch_dropout=0.5,
        in_channel=6,

        model_type="",
        load_rgb=False,
        load_pretrained=False,
        ckpt_path="/path/to/your/unigs/cache/uni3d-S.pt",
    ),
    learning_rate=1e-4, # single GPU
    pts_channel=using_channel,
    forward_all=True,
    loss_weight=dict(
        text_weight=0.5,
        image_weight=0.5
    ),
    save_interval = 50,
)

log_cfg = dict(
    skip_tqdm=True,
    print_interval=10,
)

data_cfg = dict(
    dataset=dict(
        type="mvimgnet",
        data_dir="/path/to/your/gaussian-splatting/unigs/mvimgnet",
        using_channel=using_channel,
        # data_split="file_paths.json",
        data_split="sample_all.json",
    ),
    batch_size=24,
    test_batch_size=64,
    pin_memory=True,
    nw=4,
)

use_ddp_wrapper = True
find_unused_parameters = True