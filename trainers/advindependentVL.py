import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from zsrobust.utils import clip_img_preprocessing as preprocessing
from attack.pgd import attack_pgd
from dassl.evaluation import build_evaluator
from tqdm import tqdm
from autoattack import AutoAttack
import os
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.ADVIVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.ADVIVLP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.ADVIVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.ADVIVLP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def calculate_elementwise_kl_div(output1, output2):
    instance_wise_kl=F.kl_div(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1), reduction='none')
    kl_divs=torch.sum(instance_wise_kl,dim=-1)
    return kl_divs

def calculate_cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2)+1
def calculate_mae(features1, features2):
    instance_wise_mae=F.l1_loss(features1, features2,reduction='none')
    mae_loss= torch.mean(instance_wise_mae,dim=-1)
    return mae_loss
def calculate_mse(features1, features2):
    instance_wise_mse=F.mse_loss(features1, features2,reduction='none')
    mse_loss= torch.mean(instance_wise_mse,dim=-1)
    return mse_loss

def calculate_adv_loss(output_clean, output_adv, clean_image_features, adv_image_features, adv_term="cos"):
    kl_divs = calculate_elementwise_kl_div(output_adv, output_clean)
    if adv_term=="cos":
        cosine_sims = calculate_cosine_similarity(clean_image_features, adv_image_features)
        loss = torch.mean(kl_divs * cosine_sims)
    elif adv_term =="mae":
        kl_divs=kl_divs/output_clean.shape[1]
        mae_loss=calculate_mae(clean_image_features,adv_image_features)
        loss=torch.mean(kl_divs/mae_loss)
    elif adv_term =="mse":
        kl_divs=kl_divs/output_clean.shape[1]
        mse_loss=calculate_mse(clean_image_features,adv_image_features)
        loss=torch.mean(kl_divs/mse_loss)
    else:
        raise NotImplementedError
    return loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.ADVIVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.ADVIVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.ADVIVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.ADVIVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None,return_features=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        
        if return_features:
            return logits, image_features, text_features
        else:
            return logits


@TRAINER_REGISTRY.register()
class ADVIVLP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADVIVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADVIVLP.PREC == "fp32" or cfg.TRAINER.ADVIVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        self.preprocessing=preprocessing

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ADVIVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.ADVIVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:

            nat_loss = torch.tensor(0.0).to(self.device)
            adv_loss = torch.tensor(0.0).to(self.device)

            delta=attack_pgd(model, self.preprocessing, image, label, alpha=self.cfg.ATTACK.PGD.ALPHA, 
                                    attack_iters=self.cfg.ATTACK.PGD.TRAIN_ITER,epsilon=self.cfg.ATTACK.PGD.EPS,train_trades= self.cfg.ATTACK.PGD.ADV_TERM != "ce" )
                # delta=attack_pgd_with_cos(model, self.preprocessing, image, label, alpha=self.cfg.ATTACK.PGD.ALPHA, 
                #                     attack_iters=self.cfg.ATTACK.PGD.TRAIN_ITER,epsilon=self.cfg.ATTACK.PGD.EPS)
            image_adv= self.preprocessing(image + delta)
            output_adv, adv_image_features, adv_text_features= self.model(image_adv, return_features=True)
            image_clean=self.preprocessing(image)
            output_clean, clean_image_features, clean_text_features =self.model(image_clean, return_features=True)
            if self.cfg.ATTACK.PGD.ADV_TERM == "ce":
                adv_loss = F.cross_entropy(output_adv, label)
            elif self.cfg.ATTACK.PGD.ADV_TERM == "cos":
                nat_loss = F.cross_entropy(output_clean, label)
                adv_loss = calculate_adv_loss(output_clean,output_adv,clean_image_features,adv_image_features,self.cfg.ATTACK.PGD.ADV_TERM)
            else:
                raise NotImplementedError
            
            loss = nat_loss + self.cfg.ATTACK.PGD.LAMBDA_1 *adv_loss
                

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "Nat_loss": nat_loss.item(),
            "Adv_loss": adv_loss.item(),
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

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        
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


        perform_adv_test = True

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            
            # nature test
            with torch.no_grad():
                input, label = self.parse_batch_test(batch)
                output = self.model(self.preprocessing(input))

                self.evaluator.process(output, label)
                
            torch.cuda.empty_cache()
            
            if perform_adv_test:
                delta = attack_pgd(self.model, self.preprocessing, input, label, alpha=self.cfg.ATTACK.PGD.ALPHA, 
                                attack_iters=self.cfg.ATTACK.PGD.TEST_ITER, epsilon=self.cfg.ATTACK.PGD.EPS)
                tmp = self.preprocessing(input + delta)

                torch.cuda.empty_cache()
                with torch.no_grad():
                    output_adv = self.model(tmp)
                    self.evaluator_adv.process(output_adv, label)

        results = self.evaluator.evaluate()
        results_adv = {}

        if perform_adv_test:
            results_adv = self.evaluator_adv.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        if perform_adv_test:
            for k, v in results_adv.items():
                tag = f"{split}/{k}_adv"
                self.write_scalar(tag, v, self.epoch)

        if perform_adv_test:
            return list(results.values())[0], list(results_adv.values())[0]
        else:
            return list(results.values())[0]

