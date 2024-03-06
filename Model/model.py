from Model.segment_anything import sam_model_registry
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from Model.sam_lora_image_encoder_mask_decoder import _LoRA_qkv
import clip
from torch.nn.parameter import Parameter
import math

class Transpose_Layer(nn.Module):
    def __init__(self):
        super(Transpose_Layer, self).__init__()
        self.model = nn.Conv2d(in_channels=256, out_channels=1,kernel_size=1)
    def forward(self,x):
        return self.model(x)
    
class Mask_SAM(nn.Module):
    def __init__(
            self, 
            config, 
            device = 'cuda:0',
            r = 4,
            lora_layer=None,
            pretrained_path = None,
        ):
        super().__init__()
        self.device = device
        self.img_size = config['img_size']
        self.prompt_config = config['settings']
        self.sam_ckpt_path = config['sam_ckpt_path']
        self.tp_ckpt_path = config['tp_ckpt_path']

        print(self.prompt_config)
        if pretrained_path is None:
            sam = sam_model_registry['vit_b'](image_size=self.img_size,
                                          checkpoint = self.sam_ckpt_path)
        else:
            sam = sam_model_registry['vit_b'](image_size=self.img_size,
                                          checkpoint = pretrained_path)

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam.image_encoder.blocks)))
        self.r = r
        self.sam = sam.to(device)
        self.clip_model, _  = clip.load("ViT-B/32", device=device)
        self.transpose = Transpose_Layer() 
        self.transpose.load_state_dict(torch.load(self.tp_ckpt_path))
        if self.prompt_config['USE_TEXT_PROMPT']:
            self.Text_Embedding_Affine = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )
        
    def forward(self, image_embeddings, mask_prompts,alpha,x_text=None):
        assert self.prompt_config['USE_TEXT_PROMPT'] or self.prompt_config['USE_MASK_PROMPT'], "you should choose at least one of mask prompt or text prompt!"

        # CLIP
        if self.prompt_config['USE_TEXT_PROMPT']:
            x_text = list(x_text)
            text_inputs = (clip.tokenize(x_text)).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
            # CLIP affine layer
            text_features_affine = self.Text_Embedding_Affine(text_features.float())
            text_features_affine = text_features_affine.unsqueeze(1)

        # SAM prompt encoder
        if self.prompt_config['USE_MASK_PROMPT']: 
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=mask_prompts,
            )
        elif self.prompt_config['USE_TEXT_PROMPT']: # only text no mask
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
            )
            sparse_embeddings = sparse_embeddings.to(self.device).repeat(image_embeddings.shape[0],1,1)
        # print('sparse_embeddings shape: ', sparse_embeddings.shape)
        # print('dense_embeddings shape: ', dense_embeddings.shape)
        # sparse_embeddings shape:  torch.Size([20, 0, 256])
        # dense_embeddings shape:  torch.Size([20, 256, 32, 32])
        # text feature concat with sparse embedding
        if self.prompt_config['USE_TEXT_PROMPT']:
            sparse_embeddings = torch.cat([sparse_embeddings,text_features_affine], dim=1)
        
        # SAM decoder
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            alpha = alpha
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(self.img_size, self.img_size),
            original_size=(self.img_size, self.img_size),
        )
        return masks
    
    def add_lora(self):
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
        # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, self.r, bias=False)
            w_b_linear_q = nn.Linear(self.r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, self.r, bias=False)
            w_b_linear_v = nn.Linear(self.r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
    
    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def get_image_embeddings(self, x_img):
        with torch.no_grad():
            B, C, H, W = x_img.shape
            image_embeddings = self.sam.image_encoder(x_img)
            return image_embeddings
        
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        # masks = torch.sigmoid(masks)
        return masks #.squeeze(1)