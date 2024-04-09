import os
import json
import torch
from torch import nn
from PIL import Image
import numpy as np
from diffusers import (
    UniPCMultistepScheduler, MyKandinskyPriorPipeline,MyPriorTransformer,
    StableDiffusionControlNet2PromptPipeline,
    ControlNetModel,
    StableDiffusionPipelineV2,KandinskyPipeline, KandinskyPriorPipeline
)
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
)
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.multiprocessing as mp
import json
import time

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, height=1024,width=768,
                 data_path=""
                 ):
        super().__init__()
        self.height = height
        self.width = width
        self.data_path = data_path + "/test"
        self.data_pair = list(open(data_path + "/test_pairs.txt"))
        data = list()
        dirty_data = list(open(data_path+"/dirty_test_data.txt"))
        # 处理文件内部
        for i in range(len(self.data_pair)):
            flag = self.data_pair[i].find("jpg")
            path1 = self.data_pair[i][0:flag+3]
            path2 = self.data_pair[i][flag+4:-1]
            data.append(path1)
            data.append(path2)
        for i in range(len(dirty_data)):
            dirty_data[i] = dirty_data[i][:-1]
        data= sorted(list(set(data)), key=data.index)
        self.data = self.filter(data,dirty_data)
        self.clip_image_processor = CLIPImageProcessor()
        
    
    def filter(self,data,dirty_data):
        Data = list()
        for path in data:
            index = path[:5]
            if index not in dirty_data:
                Data.append(path)
        return Data
        
    def __getitem__(self, idx):
        pair = self.data[idx]
        index = pair[:-4]             
        # read image
        Isa = self.clip_image_processor(
            images=(Image.open(self.data_path + 
             "/image-parse-agnostic-v3.2/"+ index + ".png")).convert("RGB"),
             return_tensors="pt"
        ).pixel_values

        Idp = self.clip_image_processor(
            images=(Image.open(self.data_path + 
             "/image-densepose/"+ pair)).convert("RGB"),
             return_tensors="pt"
        ).pixel_values

        Ic = self.clip_image_processor(
            images=(Image.open(self.data_path+
             "/cloth/"+pair).convert("RGB")),
            return_tensors="pt"
        ).pixel_values

        Icm = self.clip_image_processor(
            images=(Image.open(self.data_path+
             "/cloth-mask/"+pair).convert("RGB")),
            return_tensors="pt" 
        ).pixel_values
        
        Iw = self.clip_image_processor(
            images=(Image.open(self.data_path+
             "/warp-cloth/"+pair).convert("RGB")),
             return_tensors="pt"
        ).pixel_values
    
        return {
            "Isa": Isa,
            "Idp": Idp,
            "Ic": Ic,
            "Icm": Icm,
            "Iw": Iw,
            "path":pair
        }
    
    def __len__(self):
        return len(self.data)


def collate_fn(data):
    Isa_ = torch.cat([example["Isa"] for example in data],dim=0)
    Idp_ = torch.cat([example["Idp"] for example in data],dim=0)
    Ic_ = torch.cat([example["Ic"] for example in data],dim=0)
    Icm_ = torch.cat([example["Icm"] for example in data],dim=0)
    Iw_ = torch.cat([example["Iw"] for example in data],dim=0)
    path_ = [example["path"] for example in data]
    return {
            "Isa_": Isa_,
            "Idp_": Idp_,
            "Ic_": Ic_,
            "Icm_": Icm_,
            "Iw_": Iw_,
            "path_":path_
    }

def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks

def main(args, rank, select_test_datas,):


    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # save path
    save_dir = "path/stage1/{}/{}_guidancescale{}_seed{}_numsteps{}/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number, args.num_inference_steps)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # prepare data aug
    clip_image_processor = CLIPImageProcessor()

    # prepare model
    model_ckpt = "path/{}/{}/mp_rank_00_model_states.pt".format(
        args.exp_name, args.weights_number)


    pipe = MyKandinskyPriorPipeline.from_pretrained(args.pretrained_model_name_or_path).to(device)
    pipe.prior= MyPriorTransformer.from_pretrained(args.pretrained_model_name_or_path, subfolder="prior", num_embeddings=3,embedding_dim=1024, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

    prior_dict = torch.load(model_ckpt, map_location="cpu")["module"]
    pipe.prior.load_state_dict(prior_dict)
    pipe.enable_xformers_memory_efficient_attention()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,subfolder="image_enocder").eval().to(device)
    


    print('====================== model load finish ===================')


    # start test
    start_time = time.time()
    number = 0
    sum_simm = 0
    for select_test_data in select_test_datas:
        number += 1

        #prepare data
        s_img_path = (args.img_path + select_test_data["source_image"].replace('.jpg', '.png'))
        t_img_path = (args.img_path + select_test_data["target_image"].replace('.jpg', '.png'))

        s_pose_path = args.pose_path + select_test_data['source_image'].replace('.jpg', '.txt')
        t_pose_path = (args.pose_path + select_test_data["target_image"].replace(".jpg", ".txt"))

        # image_pair
        s_image = Image.open(s_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        t_image = Image.open(t_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)

        s_pose = read_coordinates_file(s_pose_path).to(device).unsqueeze(1)
        t_pose = read_coordinates_file(t_pose_path).to(device).unsqueeze(1)





        clip_s_image = clip_image_processor(images=s_image, return_tensors="pt").pixel_values
        clip_t_image = clip_image_processor(images=t_image, return_tensors="pt").pixel_values


        with torch.no_grad():
            s_img_embed = (image_encoder(clip_s_image.to(device)).image_embeds).unsqueeze(1)
            target_embed = image_encoder(clip_t_image.to(device)).image_embeds




        output = pipe(
            s_embed = s_img_embed,
            s_pose = s_pose,
            t_pose = t_pose,
            num_images_per_prompt=1,
            num_inference_steps = args.num_inference_steps,
            generator = generator,
            guidance_scale = args.guidance_scale,
        )

        # save features
        feature = output[0].cpu().detach().numpy()
        if args.save_npy==1:
            np.save(save_dir + s_img_path.split("/")[-1].replace(".png", '_to_') + t_img_path.split("/")[-1].replace(".png", '.npy'),  feature)

        # computer scores
        predict_embed = output[0]

        cosine_similarities = F.cosine_similarity(predict_embed, target_embed)

        sum_simm += cosine_similarities.item()

    end_time =time.time()
    print(end_time-start_time)


    avg_simm = sum_simm/number
    with open (save_dir+'/a_results.txt', 'a') as ff:
        ff.write('number is {}, guidance_scale is {}, all averge simm is :{} \n'.format(number, args.guidance_scale, avg_simm))

    print('number is {}, guidance_scale is {}, all averge simm is :{}'.format(number, args.guidance_scale, avg_simm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a prior model of stage1 script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="path",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--img_path", type=str, default="path", help="image path", )
    parser.add_argument("--pose_path", type=str, default="path", help="pose path", )
    parser.add_argument("--json_path", type=str, default="path", help="json path", )
    parser.add_argument("--guidance_scale",type=int,default=0,help="device number",)
    parser.add_argument("--seed_number",type=int,default=42,help="device number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="device number",)
    parser.add_argument("--img_width",type=int,default=512,help="device number",)
    parser.add_argument("--img_height",type=int,default=512,help="device number",)
    parser.add_argument("--exp_name",type=str,default="baseline",help="device number",)
    parser.add_argument("--weights_number",type=int,default=100000,help="device number",)
    parser.add_argument("--save_npy", type=int, default=1, help="1:save", )
    args = parser.parse_args()
    print(args)

    # setting GPU numbers
    num_devices = torch.cuda.device_count()

    print("Using {} GPUs inference".format(num_devices))

    dataset = MyDataset(
        height = 1024,
        width = 768,
        data_path="../dataset/VITON-HD")

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle = False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )


    print('The number of test data: {}'.format(len(dataset)))

    # multi processes
    mp.set_start_method("spawn")  
    data_list = split_list_into_chunks(dataset, num_devices) 

    processes = []
    for rank in range(num_devices):
        p = mp.Process(target=main, args=(args,rank, data_list[rank], ))
        processes.append(p)
        p.start()


    for rank, p in enumerate(processes):
        p.join()








