name = 'sun_xyz'
work_dir = 'work_dirs/' + name

using_channel = 6

train_cfg = dict(
    epoch=50,
)

test_cfg = dict(
    test_gt=False,
    val_interval=10,# 10

    dataset="sunrgbd",
    task="classification",
    test_interval=10,
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
    print_interval=50,
)

data_cfg = dict(
    dataset=dict(
        type="sun",
        data_dir="/path/to/your/gaussian-splatting/unigs/sunrgbd_all",
        using_channel=using_channel,
        refine_label=True,
        # data_split="file_paths.json",
        data_split="sample_all.json",
    ),
    batch_size=24,
    pin_memory=False,
    nw=4,
)

use_ddp_wrapper = True
find_unused_parameters = True