import torch
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


class simpleTryDiffusion:
    def __init__(self,pipe,SOU,device):
        self.device = device
        self.pipe = pipe.to(device)
        self.SOU = SOU.to(device,dtype=torch.float16)
        # 更新参数
        self.pipe.unet = self.SOU.unet
   
         
    def __call__(self,cloth,mask,
                prompt=None,
                negative_prompt = None,
                guidance_scale=7.5,
                seed = -1,
                num_inference_steps=30,
                num_samples = 4,
               ):
        
        # 更新 unet self.pipe.unet = SOD.unet

        hidden_state = self.SOU.image_proj_model(cloth)
        uncon_hidden = self.SOU.image_proj_model(torch.zeros_like(cloth))
        noise = torch.randn_like(mask)
        latent = noise + mask
        
        
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
       
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality" 

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, hidden_state], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncon_hidden], dim=1)

        images = self.pipe(
            prompt_embeds =  prompt_embeds,
            negative_prompt_embeds =  negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image = latent,
            ).images
        
        return images
    



            
            












    

