import os
import time
import json
import torch
import datetime
from math import ceil
from tqdm import tqdm
from prettytable import PrettyTable

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
from constant import Constant

torch.backends.cudnn.benchmark = True

def test_gt(model, test_dataloader): # old
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

def train_epoch(model, train_dataloader, device, model_loss, optimizer,
                local_rank_flag, wirter, all_time_tick,
                print_interval, gpus, global_step, batch_size, epoch, training_data_length
                ):
    model.train()
    step = -1

    for meta_data in train_dataloader:
        # if step == 10 : break
        
        step += 1
        global_step += 1

        runing_time_tick = time.perf_counter()

        valid_mask = None
        if "valid" in meta_data.keys(): valid_mask = meta_data['valid'].to(device)

        image_features, text_features, gaussian_features, text_mask, text_embedding = \
            model(meta_data["text"], meta_data["image"].to(device), meta_data["gaussian"].to(device))
        loss, log_vars = model_loss(image_features, text_features, gaussian_features, text_mask, text_embedding, valid_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if step % print_interval == 0 and local_rank_flag:
            wirter.add_scalar("train/loss", float(loss), global_step=global_step*gpus)
            wirter.add_scalar("train/image-3d-loss", log_vars["image_3d_loss"], global_step=global_step*gpus)
            wirter.add_scalar("train/text-3d-loss", log_vars["text_3d_loss"], global_step=global_step*gpus)

            now_time_tick = time.perf_counter()
            all_time_gap = now_time_tick - all_time_tick
            runing_time_gap = now_time_tick - runing_time_tick
            all_time_tick = now_time_tick

            ave_runing_time = runing_time_gap/batch_size
            ave_all_time = all_time_gap/batch_size

            print("%s Epoch:[%d]|[%d|%d], Times:[ %.3fs| %.3fs, %.3fs], loss:%.4f, text_3d_loss:%.6f, image_3d_loss:%.6f"%(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, step, training_data_length, 
                all_time_gap, ave_all_time, ave_runing_time, float(loss), log_vars["text_3d_loss"], log_vars["image_3d_loss"]
            ),flush=True)

    return global_step
        
def log_epoch(text_accuracy, image_accuracy, total_num,
              results_tb, collect_data_list, collect_data, writer=None, epoch=0):
    results_tb_rows = [
        ["Text", str(text_accuracy.item())+"/"+str(total_num)],
        ["accuracy", round((text_accuracy/total_num).item(),4)],
        ["Image ", str(image_accuracy.item())+"/"+str(total_num)],
        ["accuracy", round((image_accuracy/total_num).item(),4)],
    ]
    for key in collect_data_list:
        results_tb_rows[0].append( str(collect_data[key][0]) + "/" + str(collect_data[key][2]) )
        results_tb_rows[1].append( str(round(collect_data[key][0]/(collect_data[key][2]+1e-8),4)) )
        results_tb_rows[2].append( str(collect_data[key][1]) + "/" + str(collect_data[key][2]) )
        results_tb_rows[3].append( str(round(collect_data[key][1]/(collect_data[key][2]+1e-8),4)) )
    results_tb.add_row(results_tb_rows[0])
    results_tb.add_row(results_tb_rows[1], divider=True)
    results_tb.add_row(results_tb_rows[2])
    results_tb.add_row(results_tb_rows[3])
    print(results_tb, flush=True)

    writer.add_scalar("test/text-3d", text_accuracy/total_num, global_step=epoch)
    writer.add_scalar("test/image-3d", image_accuracy/total_num, global_step=epoch)

@torch.no_grad()
def val_epoch(model, test_dataloader, test_text, device, local_rank_flag, skip_tqdm, 
              first_epoch_flag, collect_data_list, data_type_list, writer, epoch):
    model.eval()

    image_accuracy, text_accuracy, total_num = 0, 0, 0

    if local_rank_flag:
        if skip_tqdm and not first_epoch_flag: 
            print("\nEvaling bacth ...")
            test_dataloader_iterator = test_dataloader
        else:
            test_dataloader_iterator = tqdm(test_dataloader, desc="Evaling bacth")
    else: test_dataloader_iterator = test_dataloader

    # table
    collect_data = {key:[0,0,0] for key in collect_data_list} # text image all
    results_tb = PrettyTable(["", "3D accuracy", *collect_data_list])

    for meta_data in test_dataloader_iterator:
        images = meta_data["image"].to(device) # [b,36,3,224,224]

        batch = images.shape[0]
        image_gt_label = torch.arange(batch, device="cuda")

        if test_text is None: 
            test_text = meta_data["text"]
            text_gt_label = image_gt_label

            # text_gt_label = np.array([i for i in range(len(test_text))])
            # random.shuffle(text_gt_label)
            # test_text = np.array(test_text)[text_gt_label].tolist()
            # text_gt_label = torch.from_numpy(text_gt_label).to(image_gt_label.device)

        else:
            text_gt_label = meta_data["label_count"].to(device)

        image_label, text_label = model.module.predict(
            test_text, meta_data["image"].to(device), meta_data["gaussian"].to(device), use_mask=(test_text is None)
        )
    
        total_num += batch
        image_accuracy += torch.sum(image_gt_label==image_label)
        text_accuracy += torch.sum(text_gt_label==text_label)

        for j in range(len(data_type_list)):
            if data_type_list[j] in collect_data_list:
                collect_data[data_type_list[j]][0] += torch.sum((text_gt_label==j) * (text_gt_label==text_label)).item()
                collect_data[data_type_list[j]][2] += torch.sum(text_gt_label==j).item()
                collect_data[data_type_list[j]][1] += torch.sum((text_gt_label==j) * (image_gt_label==image_label)).item()

    if local_rank_flag:
        log_epoch(text_accuracy, image_accuracy, total_num,
                results_tb, collect_data_list, collect_data, writer, epoch)
    
    return total_num, text_accuracy

@torch.no_grad()
def test_epoch(model, test_dataloader, task, dataset, test_func, test_text, device, local_rank_flag, skip_tqdm, 
              first_epoch_flag, collect_data_list, data_type_list, writer, epoch):
    model.eval()

    if task is None: return
    test_dataloader.dataset.file_paths = test_dataloader.dataset.file_paths_copy.copy()

    if task == "retrive":
        total_num, text_accuracy = test_func(model.module, test_dataloader, device, [1,3,5,10,20,30,50], local_rank_flag, first_epoch_flag)
        writer.add_scalar("test/all-text-3d", text_accuracy/total_num, global_step=epoch)
    elif task == "classification":
        if dataset == "objaverse":
            total_num, text_accuracy = val_epoch(model, test_dataloader, test_text, device, local_rank_flag, skip_tqdm, 
                        first_epoch_flag, collect_data_list, data_type_list, writer, epoch)
            writer.add_scalar("test/all-text-3d", text_accuracy/total_num, global_step=epoch)

def train_loop(
        model,
        model_loss,
        cfg,
        train_dataloader,
        test_dataloader,
        optimizer,
        batch_size,
        lr_scheduler,
        local_rank="-1",
        validate=True,
        debug=False,
):
    # geting param from cfg
    EPOCH = cfg["train_cfg"]["epoch"]
    print_interval = cfg["log_cfg"]["print_interval"]
    save_interval = cfg["model_cfg"]["save_interval"]
    work_dir = cfg["work_dir"]
    skip_tqdm = cfg["log_cfg"].get("skip_tqdm", False)

    ## training
    global_step = 0
    project_root = os.path.join(work_dir, "log")
    writer = SummaryWriter(project_root)

    ## testing
    val_interval = cfg["test_cfg"]["val_interval"]
    test_interval = cfg["test_cfg"]["test_interval"]
    task = cfg["test_cfg"]["task"]
    dataset = cfg["test_cfg"].get("dataset", cfg["data_cfg"]["dataset"]["type"])

    constant = Constant(task, dataset)
    test_text = constant.get_testing_text()
    data_type_list, collect_data_list, test_func = constant.data_type_list, constant.collect_data_list, constant.test_func

    gpus = len(cfg.gpu_ids)
    local_rank_flag = (local_rank == "0")
    training_data_length = ceil(len(train_dataloader.dataset)/batch_size)//gpus
    device = "cuda" if not debug else "cpu"

    if local_rank_flag: print(cfg,flush=True)

    for epoch in range(EPOCH):
        all_time_tick = time.perf_counter()
        if local_rank_flag: print("\nTraining Epoch %d"%(epoch))

        global_step = train_epoch(model, train_dataloader, device, model_loss, optimizer, local_rank_flag, writer,
                    all_time_tick, print_interval, gpus, global_step, batch_size, epoch, training_data_length)
        
        # saving
        if (epoch + 1) % save_interval == 0 and local_rank_flag:
            model_saving_path = os.path.join(work_dir, "model_epoch"+str(epoch+1)+".pth")
            print("Saving model at ", model_saving_path)
            # torch.save(model.module, model_saving_path)
            torch.save(model.module.pointnet.state_dict(), model_saving_path)

        # val
        first_epoch_flag = epoch==0
        if validate and ((epoch+1) % val_interval == 0 or first_epoch_flag):

            if task == "classification" and dataset == "objaverse":
                test_dataloader.dataset.file_paths = test_dataloader.dataset.file_paths_copy[::50]

            val_epoch(model, test_dataloader, test_text, device, local_rank_flag, skip_tqdm, 
                      first_epoch_flag, collect_data_list, data_type_list, writer, epoch)
        
        # test
        # if validate and (epoch+1) % test_interval == 0:
        onjaverse_lvis_flag = dataset == "objaverse" and task == "classification"
        first_epoch_flag = first_epoch_flag and not onjaverse_lvis_flag
        if validate and ((epoch+1) % test_interval == 0 or first_epoch_flag):
            test_epoch(model, test_dataloader, task, dataset, test_func, test_text, device, local_rank_flag, skip_tqdm, 
                      first_epoch_flag, collect_data_list, data_type_list, writer, epoch)
        
        # after a epoch
        lr_scheduler.step()
        # exit()

    # last saving
    if local_rank_flag and (epoch + 1) % save_interval != 0:
        model_saving_path = os.path.join(work_dir, "model_epoch_last.pth")
        print("Saving model at ", model_saving_path)
        torch.save(model.module.pointnet.state_dict(), model_saving_path)

    if local_rank == 0 : writer.close()

def train_model(model,
                model_loss,
                datasets,
                cfg,
                distributed=False,
                validate=False,
                debug=False):
    
    if debug:
        train_sampler = None
        setattr(model, "module", model)
    else:
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            model = DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters
            )
            train_sampler = DistributedSampler(datasets[0])
            test_sampler = DistributedSampler(datasets[1])
        else:
            model = DataParallel(model, device_ids=cfg.gpu_ids)
            train_sampler, test_sampler = None, None

    # dataloader
    batch_size = cfg["data_cfg"]["batch_size"]
    num_works = cfg["data_cfg"]["nw"]
    pin_memory = cfg["data_cfg"]["pin_memory"]
    train_loader = DataLoader(datasets[0], batch_size=batch_size, num_workers=num_works, pin_memory=pin_memory, sampler=train_sampler)
    test_loader = DataLoader(datasets[1], batch_size=batch_size, num_workers=num_works, pin_memory=pin_memory)

    # optimizer
    optimizer = model.module.configure_optimizers()
    # lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200], gamma=0.8)
    lr_scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.5)
    # 0.8  0.64  0.512  0.4096  0.32768 | 0.262  0.209  0.167  0.134  0.107

    # train 
    train_loop(
        model, model_loss, cfg, train_loader, test_loader, optimizer, batch_size, lr_scheduler,
        local_rank=os.environ['RANK'], 
        validate=validate,
        debug=debug,
    )
