import torch
from PIL import Image
from transformers import CLIPImageProcessor



def PriorCollate_fn(data):
    Isa_ = torch.cat([example["Isa"] for example in data],dim=0)
    Idp_ = torch.cat([example["Idp"] for example in data],dim=0)
    Ic_ = torch.cat([example["Ic"] for example in data],dim=0)
    Icm_ = torch.cat([example["Icm"] for example in data],dim=0)
    Iw_ = torch.cat([example["Iw"] for example in data],dim=0)
    return {
            "Isa_": Isa_,
            "Idp_": Idp_,
            "Ic_": Ic_,
            "Icm_": Icm_,
            "Iw_": Iw_
    }



class PriorImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=""):
        super().__init__()
        self.data_path = data_path + "/train"
        self.data_pair = list(open(data_path + "/train_pairs.txt"))
        data = list()
        dirty_data = list(open("dirty_train_data.txt"))
        # 处理文件内部
        for i in range(len(self.data_pair)):
            flag = self.data_pair[i].find("jpg")
            path1 = self.data_pair[i][0:flag+3]
            path2 = self.data_pair[i][flag+4:-1]
            data.append(path1)
            data.append(path2)
        # 去掉脏数据
        for i in range(len(dirty_data)):
            dirty_data[i] = dirty_data[i][:-1]
        data = list(set(data))
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
            "Iw": Iw
        }
    
    def __len__(self):
        return len(self.data)
