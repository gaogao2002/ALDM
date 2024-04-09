from PIL import Image
import torch
from torchvision import transforms
import random
from transformers import CLIPImageProcessor


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, size=256,
                 cloth_drop_rate=0.05,
                 mask_drop_rate=0.05,
                 data_path="",
                 ):
        super().__init__()

        self.size = size
        # 给予在训练时一定元素缺失，增强模型的生成能力
        self.cloth_drop_rate = cloth_drop_rate
        self.mask_drop_rate = mask_drop_rate
        self.data_path = data_path + "/train"
        self.data_pair = list(open(data_path + "/train_pairs.txt"))
        self.data = list()
        # 处理文件内部
        for i in range(len(self.data_pair)):
            flag = self.data_pair[i].find("jpg")
            path1 = self.data_pair[i][0:flag+3]
            path2 = self.data_pair[i][flag+4:-1]
            self.data.append(path1)
            self.data.append(path2)
        self.transform = transforms.Compose([
            transforms.Resize(
                self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        pair = self.data[idx]
        # read image
        warp_cloth = Image.open(self.data_path+"/warp-cloth/"+pair).convert("RGB")
        people = self.transform(
            (Image.open(self.data_path+"/image/"+pair)).convert("RGB")
        )
        mask = self.transform(
            (Image.open(self.data_path +
             "/agnostic-v3.2/"+pair)).convert("RGB")
        )

        warp_cloth = self.clip_image_processor(
            images=warp_cloth, return_tensors="pt").pixel_values

        # drop
        rand_num = random.random()
        if rand_num < self.cloth_drop_rate:
            warp_cloth = torch.zeros_like(warp_cloth)
        elif rand_num < (self.cloth_drop_rate + self.mask_drop_rate):
            mask = torch.zeros_like(mask)

        return {
            "warp_cloth": warp_cloth,
            "people": people,
            "mask": mask,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    peoples = torch.stack([example["people"] for example in data])
    warp_clothes = torch.cat([example["warp-cloth"] for example in data], dim=0)
    masks = torch.stack([example["mask"] for example in data])
    return {
        "warp_clothes": warp_clothes,
        "peoples": peoples,
        "masks": masks
    }
