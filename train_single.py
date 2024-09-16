import os
import torch
from unigs.pointnet.get_model import UniGS
from tqdm import tqdm
from lib.ABO_dataset import get_ABO_dataloader, get_ABO_val_test_dataloader
from lib.SUN_dataset import get_SUN_dataloader
import time
import argparse

from torch.utils.tensorboard import summary
from torch.utils.tensorboard import SummaryWriter


def test_gt(model, test_dataloader):
    ## Testing on gt
    print("Evaling on gt ...", flush=True)
    accuracy, total_num, accuracy_any = 0, 0, []
    for meta_data in tqdm(test_dataloader):
        images = meta_data["image"].cuda() # [b,36,3,224,224]

        accuracy_any.append(0)

        for i in range(images.size(1)):

            image_label = model.predict_clip(
                meta_data["clip"], images[:,i]
            )

            batch = images.shape[0]
            gt_label = torch.arange(batch, device="cuda")
        
            total_num += batch
            this_accuracy = torch.sum(gt_label==image_label)
            accuracy += this_accuracy
            if this_accuracy > 0: accuracy_any[-1] = 1

    print(flush=True)
    print("GT Text-Image accuracy: All: %.4f | Any: %.4f"%(accuracy/total_num, sum(accuracy_any)/len(accuracy_any) ), flush=True)
    print(flush=True)

def train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        batch_size,
        EPOCH,
        project_root,
        print_interval = 10,
        test_interval = 5,
):
    ## training
    global_step = 0
    wirter = SummaryWriter(project_root)

    for epoch in range(EPOCH):
        step = 0
        model.train()
        for meta_data in train_dataloader:
            begin = time.perf_counter()
            loss, log_vars = model(
                meta_data["text"], meta_data["image"].cuda(), meta_data["gaussian"].cuda()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            global_step += 1
            if step % print_interval == 0:
                end = time.perf_counter()
                wirter.add_scalar("train/loss", float(loss), global_step=global_step)
                wirter.add_scalars("train/log_vars", log_vars, global_step=global_step)
                print("Epoch:[%d]|[%d], Times:%.4fs : loss:%.4f, text_3d_loss:%.4f, image_3d_loss:%.4f"%(
                    epoch, step, (end-begin)/batch_size, float(loss), log_vars["text_3d_loss"], log_vars["image_3d_loss"]
                ),flush=True)
        
        if (epoch+1) % test_interval == 0:
            with torch.no_grad():
                model.eval()
                print("Evaling...", flush=True)
                image_accuracy, text_accuracy, total_num = 0, 0, 0
                for meta_data in tqdm(test_dataloader):
                    images = meta_data["image"].cuda() # [b,36,3,224,224]

                    for i in range(images.size(1)):
                        image_label, text_label = model.predict(
                            meta_data["text"], images[:,i], meta_data["gaussian"].cuda()
                        )

                        batch = images.shape[0]
                        gt_label = torch.arange(batch, device="cuda")
                    
                        total_num += batch
                        image_accuracy += torch.sum(gt_label==image_label)
                        text_accuracy += torch.sum(gt_label==text_label)

                print(flush=True)
                print("Image accuracy: %.4f"%(image_accuracy/total_num), flush=True)
                print("Text accuracy: %.4f"%(text_accuracy/total_num), flush=True)
                print(flush=True)
            
                wirter.add_scalar("test/text-3d", text_accuracy/total_num, global_step=global_step)
                wirter.add_scalar("test/image-3d", image_accuracy/total_num, global_step=global_step)

    wirter.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name',type=str,default="test")
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=24)
    parser.add_argument('--test_gt',type=bool,default=False)
    parser.add_argument('--print_interval',type=int,default=10)
    parser.add_argument('--pts_channel',type=int,default=8)
    parser.add_argument('--test_interval',type=int,default=5)
    parser.add_argument('--text_weight',type=float,default=0.5)
    parser.add_argument('--image_weight',type=float,default=0.5)
    parser.add_argument('--dataset',type=str,default="sun")
    parser.add_argument('--gpu_id',type=str,default="0")
    opt = parser.parse_args()

    # log
    experiment_name = opt.name
    log_root = "/path/to/your/unigs/log"
    project_root = os.path.join(log_root, experiment_name)
    os.makedirs(project_root, exist_ok=True)

    # train
    if opt.dataset.lower() == "sun":
        data_dir = "/path/to/your/gaussian-splatting/unigs/sunrgbd"

        train_dataloader = get_SUN_dataloader(data_dir,opt.batch_size,4)
        test_dataloader = get_SUN_dataloader(data_dir,opt.batch_size,4,is_train=False)
    elif opt.dataset.lower() == "abo":
        guassian_data_path = "/path/to/your/gaussian-splatting/output/ABO"
        image_data_path = "/path/to/your/ABO/render_images_256"
        text_data_path = "/path/to/your/ABO/ABO_prompt_compile.json"

        train_dataloader = get_ABO_dataloader(guassian_data_path,image_data_path,text_data_path,opt.batch_size,4,camera_key=["up"])
        test_dataloader = get_ABO_val_test_dataloader(guassian_data_path,image_data_path,text_data_path,opt.batch_size,4,camera_key=["up"])

    unigs = UniGS(
        clip_model="/path/to/your/unigs/cache/ViT-B-16.pt",
        pointnet_model="pointnet2_cls_msg",
        learning_rate=1e-3,
        text_weight=opt.text_weight,
        image_weight=opt.image_weight,
        pts_channel=opt.pts_channel,
    ).cuda()
    optimizer = unigs.configure_optimizers()

    if opt.test_gt: test_gt(unigs, test_dataloader)

    train(unigs, train_dataloader, test_dataloader, optimizer, opt.batch_size, opt.epoch, project_root, 
          opt.print_interval, opt.test_interval)

if __name__ == "__main__":
    main()