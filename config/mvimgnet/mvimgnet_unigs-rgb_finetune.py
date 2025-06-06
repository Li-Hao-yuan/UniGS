name = 'mvimgnet_clip3-rgb'
work_dir = 'work_dirs/mvimgnet_clip3-rgb'
using_channel = 14
train_cfg = dict(epoch=50)
test_cfg = dict(
    test_gt=False,
    val_interval=5,
    dataset='mvimgnet',
    task='retrive',
    test_interval=5)
model_cfg = dict(
    clip_model_path='/path/to/your/ViT-B-16.pt',
    pointnet_model='uni3D',
    model_setting=dict(
        pc_model='eva02_small_patch14_224',
        pretrained_pc=None,
        drop_path_rate=0.2,
        pc_feat_dim=384,
        embed_dim=1024,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        patch_dropout=0.5,
        in_channel=6,
        model_type='parallel',
        load_rgb=True,
        load_pretrained=True,
        ckpt_path='/path/to/your/uni3d-S.pt'),
    learning_rate=0.0001,
    pts_channel=14,
    forward_all=True,
    loss_weight=dict(text_weight=0.5, image_weight=0.5),
    save_interval=50,
    resume_from=
    '/path/to/your/model/pretrained/on/objaverse/model_epoch15.pth'
)
log_cfg = dict(skip_tqdm=True, print_interval=10)
data_cfg = dict(
    dataset=dict(
        type='mvimgnet',
        data_dir='/path/to/your/gaussian-splatting/clip3/mvimgnet',
        using_channel=14,
        data_split='sample_all.json'),
    batch_size=20,
    test_batch_size=64,
    pin_memory=True,
    nw=4)
use_ddp_wrapper = True
find_unused_parameters = True
gpu_ids = [0]
