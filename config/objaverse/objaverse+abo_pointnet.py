name = 'objaverse+abo_pointnet'
work_dir = 'work_dirs/' + name

using_channel = 6

train_cfg = dict(
    epoch=15,
)

test_cfg = dict(
    test_gt=False,
    val_interval=3,# 10

    dataset="abo",
    task="classification",
    test_interval=3,
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
        image_weight=0.5,
    ),
    save_interval = 15,
)

log_cfg = dict(
    skip_tqdm=True,
    print_interval=50,
)

data_cfg = dict(
    dataset=dict(
        type="objaverse_sun",
        data_dir="/path/to/your/gaussian-splatting/unigs/objaverse_all",
        using_channel=using_channel,
        data_split="sample_abo_all.json",
    ),
    batch_size=28,
    test_batch_size=64,
    pin_memory=False,
    nw=4,
)

use_ddp_wrapper = True
find_unused_parameters = True