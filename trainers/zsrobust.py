import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from zsrobust.clip import clip as Clip
from zsrobust.models.prompters import PadPrompter
from zsrobust.models.prompters import TokenPrompter
from zsrobust.models.model import *
from zsrobust.utils import clip_img_preprocessing as preprocessing
from attack.pgd import attack_pgd
from tqdm import tqdm
from autoattack import AutoAttack
import os
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    add_prompt_len = cfg.TRAINER.ZSROBUST.ADD_PROMPT_SIZE

    model, preprocess = Clip.load(backbone_name, device="cpu", jit=False, prompt_len=add_prompt_len)

    return model


def tokenized_prompt(cfg, classnames):
    # ADD TOKENIZED PROMPT
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        text_prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {text_prompts}")
        prompts_token = torch.cat([Clip.tokenize(p) for p in text_prompts])
        
        return prompts_token
    
    

class CustomCLIP(nn.Module):
    def __init__(self,cfg, clip_model,classnames):
        super().__init__()
        self.clip_model=clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.tokenized_prompt = tokenized_prompt(cfg, classnames=classnames)

    def forward(self, image,prompt_token=None):
        if prompt_token is not None:
            bs = image.size(0)
            prompt_token = prompt_token.repeat(bs, 1, 1)

        img_embed, scale_text_embed = self.clip_model(image, self.tokenized_prompt, prompt_token)
        logits_per_image = img_embed @ scale_text_embed.t()
        logits_per_text = scale_text_embed @ img_embed.t()
        return logits_per_image


@TRAINER_REGISTRY.register()
class zsrobust(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ZSROBUST.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.ZSROBUST.PREC == "fp32" or cfg.TRAINER.ZSROBUST.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, clip_model,classnames)
        self.model.eval()
        
        #ADD PROMPTER
        self.prompter = PadPrompter(cfg).to(self.device)
        self.add_prompter = TokenPrompter(cfg.TRAINER.ZSROBUST.ADD_PROMPT_SIZE).to(self.device)

        #ADD NORMALIZE
        self.preprocessing=preprocessing

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompter" not in name:
                param.requires_grad_(False)

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.tokenized_prompt = self.model.tokenized_prompt.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(list(self.prompter.parameters()) + list(self.add_prompter.parameters()), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompter", self.prompter,  self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.prompter=nn.parallel(self.prompter)
            self.add_prompter=nn.parallel(self.add_prompter)
            

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        self.prompter.train()
        self.add_prompter.train()
        
        prec = self.cfg.TRAINER.ZSROBUST.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            delta=attack_pgd(self.model,self.preprocessing,image,label, self.prompter,self.add_prompter, 
                             self.cfg.ATTACK.PGD.ALPHA,self.cfg.ATTACK.PGD.TRAIN_ITER,self.cfg.ATTACK.PGD.EPS)
            tmp = self.preprocessing(image + delta) 
            prompted_images = self.prompter(tmp)
            token_prompter = self.add_prompter()
            
            output,_ = self.model(prompted_images ,token_prompter)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = "prompter"

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        
        model_path = osp.join(directory, names, model_file)

        if not osp.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        add_prompter_state_dict=checkpoint['add_prompter']
        epoch = checkpoint["epoch"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format(names, model_path, epoch))
        # set strict=False
        self.prompter.load_state_dict(state_dict, strict=False)
        
        print("Loading weights to add prompter")
        self.add_prompter.load_state_dict(add_prompter_state_dict,strict=False)
    
    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        save_checkpoint(
                {
                    "state_dict": self.prompter.state_dict(),
                    "add_prompter": self.add_prompter.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.sched.state_dict(),
                    "val_result": val_result
                },
                osp.join(directory, "prompter"),
                is_best=is_best,
                model_name=model_name,
            )
    
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.prompter.eval()
        self.add_prompter.eval()
        
        self.evaluator.reset()
        self.evaluator_adv = build_evaluator(self.cfg, lab2cname=self.lab2cname)
        self.evaluator_adv.reset()
        torch.cuda.empty_cache()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set --Adversary")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                input, label = self.parse_batch_test(batch)
                prompt_token = self.add_prompter()
                # output = self.model_inference(input)
                output,_ = self.model(self.preprocessing(input), prompt_token)
                self.evaluator.process(output, label)
                
            torch.cuda.empty_cache()

            delta=attack_pgd(self.model,self.preprocessing,input,label, self.prompter,self.add_prompter, 
                             self.cfg.ATTACK.PGD.ALPHA,self.cfg.ATTACK.PGD.TEST_ITER,self.cfg.ATTACK.PGD.EPS)
            tmp= self.preprocessing(input + delta)

            torch.cuda.empty_cache()
            with torch.no_grad():
                # output_adv=self.model_inference(input_adv)
                output_adv, _ =self.model(tmp, prompt_token)
                self.evaluator_adv.process(output_adv, label)


        results = self.evaluator.evaluate()
        results_adv = self.evaluator_adv.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        for k, v in results_adv.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0],list(results_adv.values())[0]
    
 