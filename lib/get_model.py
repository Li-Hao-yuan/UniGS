import os,sys

import torch
import torch.nn as nn
import clip
import numpy as np
import torch.distributed as dist
import open_clip

from uni3d.uni3d import Uni3D, create_uni3d
from pointnet.pointnet2_cls_msg import get_model as get_model_pointnet2_msg
from pointnet.pointnet2_cls_ssg import get_model as get_model_pointnet2_ssg
from pointnet.pointnet_cls import get_model as get_model_pointet

from pointnet.pointnet2_my import get_model as get_model_pointnet2_my

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

#####

class UniGS(nn.Module):
    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
        # _features_rest: [1024,15,sh_dgree]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]
    
    pointnet_cls = {
        "pointnet2_cls_msg":get_model_pointnet2_msg, 
        "pointnet2_cls_ssg":get_model_pointnet2_ssg, 
        "pointnet_cls":get_model_pointet,
        "uni3D":Uni3D,

        "pointnet2_my":get_model_pointnet2_my,
    }

    def __init__(self,
                 clip_model,
                 pointnet_model,
                 forward_all=True,
                 pts_channel=8,

                 learning_rate=1e-3,
                 model_setting=None,
                 device="cuda:0"
                 ):
        super(UniGS, self).__init__()

        assert pointnet_model in ["pointnet2_cls_msg", "pointnet2_cls_ssg", "pointnet_cls", "uni3D", "pointnet2_my"]

        scratch = model_setting.get("scratch", False)
        if scratch or False:
            self.clip, _, _ = open_clip.create_model_and_transforms(model_name="EVA02-E-14-plus", 
                                                                     pretrained="/path/to/your/unigs/cache/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/open_clip_pytorch_model.bin") # laion2b_s9b_b144k
            self.clip.to(device)
        else: self.clip, _ = clip.load(clip_model, device=device)

        # self.clip, _, _ = open_clip.create_model_and_transforms(model_name="EVA02-B-16", 
        #                                                              pretrained="/path/to/your/unigs/cache/eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin") # laion2b_s9b_b144k
        # self.clip.to(device)

        if pointnet_model == "uni3D":
            load_pretrained = model_setting.get("load_pretrained", False)
            model_type = model_setting.get("model_type", "")
            load_rgb = model_setting.get("load_rgb", False)
            self.pointnet = create_uni3d(model_setting, load_pretrained, model_type, load_rgb, scratch)
        else:
            self.pointnet = self.pointnet_cls[pointnet_model](num_class=512, pts_channel=pts_channel, forward_all=forward_all, model_setting=model_setting)

        self.forward_all = forward_all
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.learning_rate = learning_rate
    
    def configure_optimizers(self):
        lr = self.learning_rate

        parameters = self.pointnet.get_parameters()

        opt = torch.optim.AdamW(list(parameters), lr=lr)
        return opt

    def simple_loss(self, feature_3d, feature_other, text_mask=None, temperature=0.07):
        batch = feature_3d.shape[0]

        feature_3d = feature_3d / feature_3d.norm(dim=1, keepdim=True).to(torch.float32)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * feature_3d @ feature_other.t()
        if text_mask is not None:
            logits_per_image[text_mask] = 0
        logits_per_text = logits_per_image.t()

        labels = torch.arange(batch, device="cuda")
        loss_image = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_text = torch.nn.functional.cross_entropy(logits_per_text, labels)

        return (loss_image+loss_text)/2

    def get_text_mask(self, text, batch):
        text_mask = torch.zeros((batch,batch), dtype=torch.bool)
        for i in range(batch):
            for j in range(batch):
                text_mask[i][j] = (text[i] == text[j])
            text_mask[i][i] = False
        return text_mask

    def predict_clip(self, text, image):
        text = clip.tokenize(text).cuda()
        text_features = self.clip.encode_text(text).to(torch.float32) # [2,512]
        image_features = self.clip.encode_image(image).to(torch.float32) # [2, 512]
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_image = torch.softmax(logits_per_image, dim=-1)

        image_label = torch.argmax(logits_per_image, dim=-1)

        return image_label

    def predict(self, text, image, guassian, use_mask=True, return_probability=False):

        image_features, text_features, gaussian_features, text_mask, _ = self.forward(text, image, guassian)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * gaussian_features @ image_features.t()
        logits_per_text = logit_scale * gaussian_features @ text_features.t()
        if use_mask: logits_per_text[text_mask] = 0

        image_label = torch.argmax(logits_per_image, dim=-1)
        text_label = torch.argmax(logits_per_text, dim=-1)

        if return_probability: return logits_per_image, logits_per_text
        return image_label, text_label
    
    def encode_text(self, text):
        return self.clip.encode_text(text)

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_gaussian(self, gaussian):
        return self.pointnet(gaussian, forward_all=self.forward_all)

    def forward(self, text, image, guassian):
        text_mask = self.get_text_mask(text, len(text)).cuda()
        text = clip.tokenize(text).cuda()

        text_features = self.encode_text(text).to(torch.float32) # [2,512]
        image_features = self.encode_image(image).to(torch.float32) # [2, 512]
        gaussian_features, _ = self.encode_gaussian(guassian) # [2, 512]

        # normalized features
        image_features = nn.functional.normalize(image_features, dim=1, p=2)
        text_features = nn.functional.normalize(text_features, dim=1, p=2)
        gaussian_features = nn.functional.normalize(gaussian_features, dim=1, p=2)

        return image_features, text_features, gaussian_features, text_mask, text


class UniGS_loss(nn.Module):
    def __init__(self,
            text_weight=0.5,
            image_weight=0.5,
            multi_gpu_loss=True,
            buffer_loss=False):
        super(UniGS_loss, self).__init__()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.multi_gpu_loss = multi_gpu_loss

        self.gaussian_buffer = []
        self.image_buffer = []
        self.text_buffer, self.text_mask_buffer, self.text_embedding_buffer = [], [], []
        self.buffer_length = 10
        self.buffer_loss = buffer_loss
    
    def gather_text_mask(self, text_mask, text_embedding):
        batch, all_batch = text_mask.shape[0], text_mask.shape[1]
        section = [get_rank()*batch, (get_rank()+1)*batch-1]

        for i in range(batch):
            for j in range(all_batch):
                if j >= section[0] and j <= section[1]: continue
                text_mask[i][j] = text_embedding[i+section[0]].equal(text_embedding[j])
        return text_mask

    def press_buffer(self, features, buffer_type):
        buffer_dict = {
            "text": self.text_buffer,
            "text_mask": self.text_mask_buffer,
            "text_embedding":self.text_embedding_buffer,
            "gaussian": self.gaussian_buffer,
            "image": self.image_buffer,
        }
        selected_buffer = buffer_dict[buffer_type]
        if len(selected_buffer) >= self.buffer_length: selected_buffer.pop(0)
        selected_buffer.append(features.clone().detach())


    def forward(self, image_features, text_features, gaussian_features, text_mask, text_embedding, valid_mask=None):
        # gaussian_features_all: [batch*rank, ...]
        # text_embedding_all: [5*2, 77]

        batch = image_features.shape[0]
        logit_scale = self.logit_scale.exp()
        
        if self.multi_gpu_loss:
            gaussian_features_all, text_features_all, image_features_all, text_mask_all, text_embedding_all = \
                all_gather_batch([gaussian_features, text_features, image_features, text_mask, text_embedding])

            labels = torch.arange(batch, device="cuda") + get_rank()*batch
        else:
            gaussian_features_all, text_features_all, image_features_all, text_mask_all, text_embedding_all = \
                    gaussian_features, text_features, image_features, text_mask, text_embedding

            labels = torch.arange(batch, device="cuda")
        
        if self.buffer_loss:
            if len(self.gaussian_buffer)==0:
                self.press_buffer(text_features_all, "text")
                self.press_buffer(text_mask_all, "text_mask")
                self.press_buffer(text_embedding_all, "text_embedding")
                self.press_buffer(gaussian_features_all, "gaussian")
                self.press_buffer(image_features_all, "image")
            else:

                text_features_all_copy = text_features_all.clone()
                gaussian_features_all_copy = gaussian_features_all.clone()
                image_features_all_copy = image_features_all.clone()
                text_mask_all_copy = text_mask_all.clone()
                text_embedding_all_copy = text_embedding_all.clone()

                text_features_all = torch.concat((text_features_all, torch.cat(self.text_buffer))).requires_grad_()
                gaussian_features_all = torch.concat((gaussian_features_all, torch.cat(self.gaussian_buffer))).requires_grad_()
                image_features_all = torch.concat((image_features_all, torch.cat(self.image_buffer))).requires_grad_()

                text_mask_all = torch.concat((text_mask_all, torch.cat(self.text_mask_buffer)))
                text_embedding_all = torch.concat((text_embedding_all, torch.cat(self.text_embedding_buffer)))

                self.press_buffer(text_features_all_copy, "text")
                self.press_buffer(text_mask_all_copy, "text_mask")
                self.press_buffer(text_embedding_all_copy, "text_embedding")
                self.press_buffer(gaussian_features_all_copy, "gaussian")
                self.press_buffer(image_features_all_copy, "image")
        
        text_mask_all = text_mask_all.T
        if self.multi_gpu_loss: text_mask_all = text_mask_all * self.gather_text_mask(text_mask_all, text_embedding_all)
        
        # [0,1,2,3,4] -> [5,6,7,8,9] 
        logits_per_pc_text = logit_scale * gaussian_features @ text_features_all.t() # [batch,batch*rank]
        logits_per_text_pc = logit_scale * text_features @ gaussian_features_all.t()
        logits_per_pc_text[text_mask_all] = 0
        logits_per_text_pc[text_mask_all] = 0

        logits_per_pc_image = logit_scale * gaussian_features @ image_features_all.t()
        logits_per_image_pc = logit_scale * image_features @ gaussian_features_all.t()

        # if valid_mask is not None:
        #     if sum(valid_mask) > 0:
        #         loss_pc_image = nn.functional.cross_entropy(logits_per_pc_image, labels, reduce=False, ignore_index=-100)
        #         loss_image_pc = nn.functional.cross_entropy(logits_per_image_pc, labels, reduce=False, ignore_index=-100)
        #         image_3d_loss = (loss_pc_image*valid_mask + loss_image_pc*valid_mask).sum()/sum(valid_mask)/2
        #     else: image_3d_loss = torch.tensor(0., device=image_3d_loss.device)
        # else:
        #     image_3d_loss = (nn.functional.cross_entropy(logits_per_pc_image, labels) + \
        #                 nn.functional.cross_entropy(logits_per_image_pc, labels)) / 2
        image_3d_loss = (nn.functional.cross_entropy(logits_per_pc_image, labels) + \
                        nn.functional.cross_entropy(logits_per_image_pc, labels)) / 2

        # if valid_mask is not None:
        #     if sum(valid_mask) > 0:
        #         loss_pc_text = nn.functional.cross_entropy(logits_per_pc_text, labels, reduce=False, ignore_index=-100)
        #         loss_text_pc = nn.functional.cross_entropy(logits_per_text_pc, labels, reduce=False, ignore_index=-100)
        #         text_3d_loss = (loss_pc_text*valid_mask + loss_text_pc*valid_mask).sum()/sum(valid_mask)/2
        #     else: text_3d_loss = torch.tensor(0., device=image_3d_loss.device)
        # else:
        #     text_3d_loss = (nn.functional.cross_entropy(logits_per_pc_text, labels, ignore_index=-100) + \
        #             nn.functional.cross_entropy(logits_per_text_pc, labels, ignore_index=-100)) / 2 
        text_3d_loss = (nn.functional.cross_entropy(logits_per_pc_text, labels, ignore_index=-100) + \
                    nn.functional.cross_entropy(logits_per_text_pc, labels, ignore_index=-100)) / 2 

        text_3d_loss *= self.text_weight
        image_3d_loss *= self.image_weight
        loss = text_3d_loss + image_3d_loss

        log_vars = {
            "text_3d_loss":float(text_3d_loss),
            "image_3d_loss":float(image_3d_loss),
        }

        return loss, log_vars

    
        
if __name__ == "__main__":
    unigs = UniGS(
        clip_model="/path/to/your/unigs/cache/ViT-B-16.pt",
        pointnet_model="pointnet2_cls_msg",
    ).cuda()
    
    batch = 2
    text = ["a chair"] * batch
    json_info = torch.rand(batch,16).cuda()
    image = torch.rand(batch, 3, 224, 224).cuda()
    gaussian = torch.rand(batch, 59, 1024).cuda()

    loss = unigs(text, image, json_info, gaussian)
    print(loss)

    predict = unigs.predict(text, image, gaussian)
    print(predict)
