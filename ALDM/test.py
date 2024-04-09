# import os
import argparse
from src.inference import simpleTryDiffusion
from diffusers import DDPMScheduler,AutoencoderKL,StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel
from transformers import AutoModel
import torch.nn as nn
import torch
from PIL import Image
import random
device = "cuda"
import os
os.environ['http_proxy']='http://127.0.0.1:7890'
os.environ['https_proxy']='http://127.0.0.1:7890'
import numpy as np
import torchvision
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import math
device="cuda"
model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
# 去掉最后的全连接层
model.fc = nn.Sequential()
model.to(device)
model.eval()

def calculate_fid_score(images1,images2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    images_t = transform(images1)
    images_g = transform(images2) 
    images_t = images_t.unsqueeze(0)
    images_g = images_g.unsqueeze(0)

    with torch.no_grad():
        act1 = model(images_t).detach().cpu().numpy()
        act2 = model(images_g).detach().cpu().numpy()
    
    mu1, sigma1 = act1.mean(axis=0), np.matrix(np.cov(act1, rowvar=False))
    mu2, sigma2 = act2.mean(axis=0), np.matrix(np.cov(act2, rowvar=False))
    #
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
 
    # calculate sqrt of product between cov, sqrtm( ) 对矩阵整体开平方
    covmean = sqrtm(sigma1.dot(sigma2))
 
    # check and correct imaginary numbers from sqrt,如果covmean中有复数，则返回true
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_model_path",
        default="./model/vae",
        type=str,
    )
    parser.add_argument(
        "--imageEncoder_model_path",
        type=str,
        default="model/dinov2-G",
    )
    parser.add_argument(
        "--scheduler_path",
        default="./model/scheduler",
        type=str,
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="dataset/VITON-HD",
        help="Test data root path",
    )
    parser.add_argument(
        "--model_train_choice",
        type = str,
        default = "all_trained/",
        help = "choose the model whether it is all_trained or just the cross-attention part"
    )
    parser.add_argument(
        "--model_trained_steps",
        default = "125000",
        type = str,
        help = "choose the best model"
    )
    parser.add_argument(
        "--num_samples",
        default = 4,
        type = int,
    )
    parser.add_argument(
        "--guidance_scale",
        default = 2.0,
        type = float,
    )
    parser.add_argument(
         "--image_reserve_path",
        type=str,
        default="./result",
        help="Test data root path",
    )
    args = parser.parse_args([])
    return args

@torch.no_grad()
def test():
    args = parse_args()
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.scheduler_path)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path)
    image_encoder = AutoModel.from_pretrained(args.image_encoder_path)
        
    # load SD pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )   
  
    flag = args.model_train_choice
    success_step = args.model_trained_steps
    model_path = "./simpleOnUnet/"+flag+str(success_step)+"/mp_rank_00_model_states.pt"
    sd_model_dict = torch.load(model_path)["module"]

    # load sd_model
    sd_model = simpleTryDiffusion(pipe,sd_model_dict,image_encoder,device)

    
    # 先获得pairs列表
    data_pair = list(open(args.data_root_path+"/test_pairs.txt"))
    data = list()

    # 处理文件内部
    for i in range(len(data_pair)):
        flag = data_pair[i].find("jpg")
        path1 = data_pair[i][0:flag+3]
        path2 = data_pair[i][flag+4:-1]
        data.append(path1)
        data.append(path2) 
    
    # read image
    for i,path in enumerate(data):
        people = Image.open(args.data_root_path + "/test/image/"+path).convert("RGB")

        # 将真实图像放入real文件夹
        people = people.resize((256, 256))
        people.save(args.image_reserve_path + "/real/" + str(i) + ".png")
        
        # 生成图
        cloth = Image.open(args.data_root_path + "/test/cloth/"+path).convert("RGB")
        mask = Image.open(args.data_root_path + "/test/agnostic-v3.2/"+path).convert("RGB")

        images = sd_model(cloth=cloth,mask=mask,seed=42,num_samples=args.num_samples,guidance_scale=args.guidance_scale)
        
        fid_list = list()

        # 从中挑选出fid与原图最小的
        for image in images:
            fid_list.append(calculate_fid_score(people,image.convert("RGB")))
        
        min_index = fid_list.index(min(fid_list))
        image = images[i]
        image.save(args.image_reserve_path + "/generate/" + str(i) + ".png")

if __name__ == "__main__":
    test()


                            


