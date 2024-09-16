name = 'mvimgnet_clip2-14'
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
    pointnet_model="pointnet2_cls_msg",
    model_setting=dict(
        load_pretrained=False
    ),
    learning_rate=1e-4,
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
    batch_size=28,
    test_batch_size=64,
    pin_memory=True,
    nw=4,
)

use_ddp_wrapper = True
find_unused_parameters = True