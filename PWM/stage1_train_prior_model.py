import logging
import os
from typing import Iterable, Optional
from packaging import version
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, MyPriorTransformer
from diffusers.optimization import get_scheduler
from transformers import  CLIPImageProcessor,CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, is_wandb_available
from stage1_dataset import PriorCollate_fn, PriorImageDataset
import torch
from PIL import Image
import argparse
import random


logger = get_logger(__name__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")



# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14


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

 
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss
    
class PriorImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 Isa_img_drop_rate=0.0,
                 Idp_img_drop_rate=0.0,
                 Ic_img_drop_rate=0.0,
                 Icm_img_drop_rate=0.0,
                 data_path=""):
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

        # # 3. drop
        # if random.random() < self.con1:
        #     clip_s_img = torch.zeros(clip_s_img.shape)

        # if random.random()< self.t_img_drop_rate:
        #     clip_t_img = torch.zeros(clip_t_img.shape)

        # if random.random() < self.s_pose_drop_rate:
        #     s_pose = torch.zeros(s_pose.shape)

        # if random.random() < self.t_pose_drop_rate:
        #     t_pose = torch.zeros(t_pose.shape)

        return {
            "Isa": Isa,
            "Idp": Idp,
            "Ic": Ic,
            "Icm": Icm,
            "Iw": Iw
        }
    
    def __len__(self):
        return len(self.data)


def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict= torch.load(load_dir, map_location="cpu")

    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    # TODO optimizer lr, and loss state

    weight_dict = checkpoint_state_dict["module"]
    new_weight_dict = {f"module.{key}": value for key, value in weight_dict.items()}
    model.load_state_dict(new_weight_dict)
    del checkpoint_state_dict

    return model, epoch, last_global_step


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a ControlNet training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
   
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Training data root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_layers",
        type=int
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=500000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=2000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_stage1",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError(
            "`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError(
            "`--validation_image` must be set if `--validation_prompt` is set"
        )

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError(
            "`--validation_prompt` must be set if `--validation_image` is set"
        )

    if (
            args.validation_image is not None
            and args.validation_prompt is not None
            and len(args.validation_image) != 1
            and len(args.validation_prompt) != 1
            and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )


    return args

def main(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creationw
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder").eval()
    prior_model = MyPriorTransformer.from_pretrained(args.pretrained_model_name_or_path, subfolder="prior", num_embeddings=3, embedding_dim=1024, num_layers=args.num_layers,low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    

    # Freeze vae and image_encoder
    image_encoder.requires_grad_(False)


    clip_mean = torch.tensor(-0.016)
    clip_std = torch.tensor(0.415)

    prior_model.clip_mean = None
    prior_model.clip_std = None

    accelerator.print("The Model parameters: Prior_model {:.2f}M, Image: {:.2f}M".format(
        count_model_params(prior_model), count_model_params(image_encoder)
    ))

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            prior_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        prior_model.enable_gradient_checkpointing()

    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        optimizer = torch.optim.AdamW(prior_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            prior_model.parameters(),
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2', prediction_type='sample')


    dataset = PriorImageDataset(
        data_path = args.data_root_path
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=PriorCollate_fn, batch_size=args.train_batch_size, num_workers=8,
        pin_memory=True,
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (accelerator.state.deepspeed_plugin is None or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        lr_scheduler = get_scheduler(name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps)
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(optimizer,
        warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"]["warmup_num_steps"])

    if (accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    prior_model, train_dataloader,optimizer, lr_scheduler = accelerator.prepare(prior_model,train_dataloader,optimizer, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)


    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("cloth-warped-image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")




    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        prior_model, last_epoch, last_global_step = load_training_checkpoint(
            prior_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}, global step: {last_global_step}")
        starting_epoch = last_epoch
        global_steps = last_global_step
        prior_model = prior_model
    else:
        global_steps = 0
        starting_epoch = 0
        prior_model = prior_model
    progress_bar = tqdm(range(global_steps, args.max_train_steps), initial=global_steps, desc="Steps",
                        # Only show the progress bar once on each machine.
                        disable=not accelerator.is_local_main_process, )
    myKLLoss = KLDivLoss()

    for epoch in range(starting_epoch, args.num_train_epochs):
        prior_model.train()
        train_loss = 0.0


        for step, batch in enumerate(train_dataloader):
            #TODO resume model
            with torch.no_grad():
                # condition
                img_encoder_output = (image_encoder(batch["Ic_"].to(dtype=weight_dtype)).image_embeds).unsqueeze(1)
                

                # Sample noise that we'll add to the image_embeds
                image_embeds = (image_encoder(
                    batch["Iw_"].to(dtype=weight_dtype)).image_embeds).unsqueeze(1)
                noise = torch.randn_like(image_embeds)
                
                # 添加 offset
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((image_embeds.shape[0], image_embeds.shape[1], 1), device=image_embeds.device)



                # Sample a random timestep for each image
                bsz = image_embeds.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=image_embeds.device)
                timesteps = timesteps.long()



                # normalized
                image_embeds = (image_embeds - clip_mean) / clip_std
                # noise
                noisy_x = noise_scheduler.add_noise(image_embeds, noise, timesteps)
                #Groud Truth
                target = image_embeds.squeeze(1)

                # Predict the noise residual and compute loss
                cond  = image_encoder(batch["Icm_"].to(dtype=weight_dtype)).image_embeds.unsqueeze(1)
                cond1 = image_encoder(batch["Isa_"].to(dtype=weight_dtype)).image_embeds.unsqueeze(1)
                cond2 = image_encoder(batch["Idp_"].to(dtype=weight_dtype)).image_embeds.unsqueeze(1)
                

            with accelerator.accumulate(prior_model):
                model_pred = prior_model(
                    noisy_x,
                    timestep=timesteps,
                    proj_embedding=img_encoder_output, # cloth
                    encoder_hidden_states=cond,   # cloth-mask 
                    encoder_hidden_states1=cond1, # 分割图
                    encoder_hidden_states2=cond2, # deep pose
                    attention_mask=None,
                ).predicted_image_embedding


                loss = F.mse_loss(model_pred.float(), target.float(),reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prior_model.parameters(), args.max_grad_norm)

                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_steps += 1
                accelerator.log({"train_loss": train_loss}, step=global_steps)
                train_loss = 0.0

                if global_steps % args.checkpointing_steps == 0:
                    checkpoint_model(
                        args.output_dir, global_steps, prior_model, epoch, global_steps
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            accelerator.log(logs, step=global_steps)

            if global_steps >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, prior_model, epoch, global_steps)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    main(args)
