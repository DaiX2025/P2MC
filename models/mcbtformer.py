import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ModuleParallel
from models.blocks_p import DownBlock, MissingCompletionBlock, MultimodalFusionBlock
from models.blocks import LayerNormProxy, UpBlock, OutBlock, FusionBlock
from models.utils import init_weights


def MaskModal(x, mask, num_modalities=4):
    B, K, C, H, W, D = x.size()
    y = torch.zeros_like(x)
    y[mask, ...] = x[mask, ...]
    x = y.view(B, -1, H, W, D)
    
    return torch.chunk(x, num_modalities, dim=1)


class Encoder(nn.Module):
    
    def __init__(self, img_size=128, in_chans=1, num_classes=4, embed_dims=[16, 32, 64, 128, 256]):
        super().__init__()

        self.conv_encoder1 = DownBlock(in_chans, embed_dims[0], kernel_size=3, stride=1)
        self.conv_encoder2 = DownBlock(embed_dims[0], embed_dims[1], kernel_size=3, stride=2)
        self.conv_encoder3 = DownBlock(embed_dims[1], embed_dims[2], kernel_size=3, stride=2)
        self.conv_encoder4 = DownBlock(embed_dims[2], embed_dims[3], kernel_size=3, stride=2)
        self.conv_encoder5 = DownBlock(embed_dims[3], embed_dims[4], kernel_size=3, stride=2)

        self.apply(init_weights)
        
    def forward(self, x):

        x1 = self.conv_encoder1(x)
        x2 = self.conv_encoder2(x1)
        x3 = self.conv_encoder3(x2)
        x4 = self.conv_encoder4(x3)
        x5 = self.conv_encoder5(x4)

        return x1, x2, x3, x4, x5


class MCBTFormer(nn.Module):
    
    def __init__(self, img_size=128, in_chans=1, num_classes=4, embed_dims=[16, 32, 64, 128, 256], num_heads=[1, 2, 4, 8, 8], mlp_ratios=[4, 4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=LayerNormProxy, depths=[2, 2, 2, 2, 2], 
                 missing_completion=[True, True, True, True, True], super_token_size=[8, 4, 2, 1, 1], num_agent_tokens=[128, 128, 128, 128, 128], n_iter=1, 
                 alignment_loss_type="smooth_l1", refine=True, refine_attention=True, apply_fusion=[False, False, False, True, True], n_win=[5, 5, 5, 5, 5], 
                 topk=[8, 16, 32, 64, 125], apply_attn_mask=False, apply_aggregation=[True, True, True, False, False], apply_ca=True, apply_sa=True, 
                 dw_dilation=[1, 2, 2], channel_split=[1, 3, 4], use_lpu=True, apply_trans=[True, True, True, True, True], 
                 apply_unimodal_trans=[True, True, True, True, True], apply_multimodal_trans=[True, True, True, True, True]):
        super().__init__()

        self.missing_completion = missing_completion
        self.apply_trans = apply_trans
        self.apply_unimodal_trans = apply_unimodal_trans
        self.apply_multimodal_trans = apply_multimodal_trans

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.mc_blocks = nn.ModuleList()
        self.mf_blocks = nn.ModuleList()

        cur = 0
        for i in range(len(embed_dims)):
            # MissingCompletionBlock
            mc_block = (
                MissingCompletionBlock(
                    input_size=img_size // (2 ** i), dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],qkv_bias=qkv_bias, 
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer, 
                    missing_completion=missing_completion[i], super_token_size=super_token_size[i], num_agent_tokens=num_agent_tokens[i],
                    n_iter=n_iter, alignment_loss_type=alignment_loss_type, refine=refine, refine_attention=refine_attention, use_lpu=use_lpu
                ) if apply_unimodal_trans[i] else ModuleParallel(nn.Identity(), f'mc_block{i + 1}')
            )
            self.mc_blocks.append(mc_block)
            
            # MultimodalFusionBlock
            mf_block = (
                nn.ModuleList([
                    MultimodalFusionBlock(
                        input_size=img_size, dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, 
                        qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, 
                        apply_fusion=apply_fusion[i], n_win=n_win[i], topk=topk[i], apply_attn_mask=apply_attn_mask, use_lpu=use_lpu, 
                        apply_aggregation=apply_aggregation[i], apply_ca=apply_ca, apply_sa=apply_sa, dw_dilation=dw_dilation, channel_split=channel_split,
                    ) for j in range(depths[i])
                ]) if apply_multimodal_trans[i] else ModuleParallel(nn.Identity(), f'mf_block{i + 1}')
            )
            self.mf_blocks.append(mf_block)
            
            cur += depths[i]
        
        self.apply(init_weights)
    
    def forward(self, x, mask):
        outs = [[] for _ in range(4)]
        alignment_loss = []
        
        for stage in range(5):
            x_stage = x[stage]
            size = x_stage[0].shape[2:]
            alignment_loss_stage = torch.tensor(0.0, device=mask.device)

            if self.apply_trans[stage]:
                if self.apply_unimodal_trans[stage]:
                    x_stage, alignment_loss_stage = self.mc_blocks[stage](x_stage, mask)

                if self.apply_multimodal_trans[stage]:
                    if not self.missing_completion:
                        x_stage = MaskModal(torch.stack(x_stage, dim=1), mask)
                    for blk in self.mf_blocks[stage]:
                        x_stage = blk(x_stage, mask)
                
                if x_stage[0].shape[2:] != size:
                    print(size, x_stage[0].shape)
                    x_stage = [F.interpolate(x_, size=size, mode="trilinear", align_corners=False) for x_ in x_stage]
            
            if not self.missing_completion:
                x_stage = MaskModal(torch.stack(x_stage, dim=1), mask)

            for i in range(4):
                outs[i].append(x_stage[i].contiguous())
            alignment_loss.append(alignment_loss_stage)

        return outs, alignment_loss



class Decoder_sep(nn.Module):
    
    def __init__(self, img_size=128, in_chans=1, num_classes=4, embed_dims=[16, 32, 64, 128, 256]):
        super().__init__()

        self.conv_decoder1 = UpBlock(embed_dims[1], embed_dims[0], kernel_size=3, stride=1)
        self.conv_decoder2 = UpBlock(embed_dims[2], embed_dims[1], kernel_size=3, stride=1)
        self.conv_decoder3 = UpBlock(embed_dims[3], embed_dims[2], kernel_size=3, stride=1)
        self.conv_decoder4 = UpBlock(embed_dims[4], embed_dims[3], kernel_size=3, stride=1)

        self.out = nn.Conv3d(embed_dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        self.apply(init_weights)
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        
        x4 = self.conv_decoder4(x5, x4)
        x3 = self.conv_decoder3(x4, x3)
        x2 = self.conv_decoder2(x3, x2)
        x1 = self.conv_decoder1(x2, x1)

        out = self.softmax(self.out(x1))

        return out


class Decoder_fuse(nn.Module):
    
    def __init__(self, img_size=128, in_chans=4, num_classes=4, embed_dims=[16, 32, 64, 128, 256]):
        super().__init__()

        self.fusion1 = FusionBlock(embed_dims[0]*in_chans, embed_dims[0])
        self.fusion2 = FusionBlock(embed_dims[1]*in_chans, embed_dims[1])
        self.fusion3 = FusionBlock(embed_dims[2]*in_chans, embed_dims[2])
        self.fusion4 = FusionBlock(embed_dims[3]*in_chans, embed_dims[3])
        self.fusion5 = FusionBlock(embed_dims[4]*in_chans, embed_dims[4])

        self.conv_decoder1 = UpBlock(embed_dims[1], embed_dims[0], kernel_size=3, stride=1)
        self.conv_decoder2 = UpBlock(embed_dims[2], embed_dims[1], kernel_size=3, stride=1)
        self.conv_decoder3 = UpBlock(embed_dims[3], embed_dims[2], kernel_size=3, stride=1)
        self.conv_decoder4 = UpBlock(embed_dims[4], embed_dims[3], kernel_size=3, stride=1)

        self.out2 = OutBlock(in_channels=embed_dims[1], num_classes=num_classes)
        self.out3 = OutBlock(in_channels=embed_dims[2], num_classes=num_classes)
        self.out4 = OutBlock(in_channels=embed_dims[3], num_classes=num_classes)
        self.out5 = OutBlock(in_channels=embed_dims[4], num_classes=num_classes)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.out = nn.Conv3d(embed_dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        self.apply(init_weights)
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        
        x5 = self.fusion5(x5)
        out5 = self.out5(x5)

        x4 = self.fusion4(x4)
        x4 = self.conv_decoder4(x5, x4)
        out4 = self.out4(x4)

        x3 = self.fusion3(x3)
        x3 = self.conv_decoder3(x4, x3)
        out3 = self.out3(x3)

        x2 = self.fusion2(x2)
        x2 = self.conv_decoder2(x3, x2)
        out2 = self.out2(x2)

        x1 = self.fusion1(x1)
        x1 = self.conv_decoder1(x2, x1)

        out = self.softmax(self.out(x1))

        return out, (self.up2(out2), self.up4(out3), self.up8(out4), self.up16(out5))


class Model(nn.Module):
    
    def __init__(self, img_size=128, in_chans=4, num_classes=4, embed_dims=[16, 32, 64, 128, 256], num_heads=[1, 2, 4, 8, 8], mlp_ratios=[4, 4, 4, 4, 4], 
                 depths=[1, 1, 1, 1, 1], missing_completion=[True, True, True, True, True], super_token_size=[8, 8, 4, 2, 1], num_agent_tokens=[64, 64, 64, 64, 64], 
                 n_iter=1, alignment_loss_type='smooth_l1', apply_fusion=[False, False, False, False, False], n_win=[5, 5, 5, 5, 5], topk=[1, 4, 16, 64, 125], 
                 apply_attn_mask=False, apply_aggregation=[True, True, True, True, True], apply_ca=True, apply_sa=True, dw_dilation=[1, 2, 2], 
                 channel_split=[1, 3, 4], apply_trans=[True, True, True, True, True], apply_unimodal_trans=[True, True, True, True, True], 
                 apply_multimodal_trans=[True, True, True, True, True], use_aug=True):
        super().__init__()
        
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.use_aug = use_aug
        self.is_training = False
        self.missing_completion = missing_completion
        self.alignment_loss_type = alignment_loss_type

        self.encoder = Encoder(img_size=img_size, in_chans=1, num_classes=num_classes, embed_dims=embed_dims)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.transformer = MCBTFormer(img_size=img_size, in_chans=1, num_classes=num_classes, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, depths=depths, 
                                      missing_completion=missing_completion, super_token_size=super_token_size, num_agent_tokens=num_agent_tokens, n_iter=n_iter, 
                                      alignment_loss_type=alignment_loss_type, apply_fusion=apply_fusion, n_win=n_win, topk=topk, apply_attn_mask=apply_attn_mask, 
                                      apply_aggregation=apply_aggregation, apply_ca=apply_ca, apply_sa=apply_sa, dw_dilation=dw_dilation, channel_split=channel_split, 
                                      apply_trans=apply_trans, apply_unimodal_trans=apply_unimodal_trans, apply_multimodal_trans=apply_multimodal_trans)

        self.decoder_aug = Decoder_sep(img_size=img_size, in_chans=1, num_classes=num_classes, embed_dims=embed_dims) if self.use_aug else nn.Identity()
        self.decoder_sep = Decoder_sep(img_size=img_size, in_chans=1, num_classes=num_classes, embed_dims=embed_dims)
        # self.decoder_sep = nn.ModuleList([Decoder_sep(img_size=img_size, in_chans=1, num_classes=num_classes, embed_dims=embed_dims) for _ in range(in_chans)]) if self.use_aug else nn.ModuleList([nn.Identity() for _ in range(in_chans)])
        # for param in self.decoder_aug.parameters():
        #     param.requires_grad = False
        # for param in self.decoder_sep.parameters():
        #     param.requires_grad = False

        # self.decoder_fuse = Decoder_fuse(img_size=img_size, in_chans=in_chans, num_classes=num_classes, embed_dims=embed_dims)

        # Initialize alpha
        self.alpha = nn.ParameterDict()
        for i in range(1, 2**self.in_chans):
            modality_comb = bin(i)[2:].zfill(self.in_chans)
            alpha_tensor = torch.tensor(
                [[int(b)] * self.num_classes for b in modality_comb],
                dtype=torch.float32,
                requires_grad=True
            )
            self.alpha[modality_comb] = nn.Parameter(alpha_tensor)

        self.apply(init_weights)

    def forward(self, x, mask):
        # for modality_comb, alpha_param in self.alpha.items():
        #     print(f"Alpha {modality_comb}: grad={alpha_param.grad}, value={alpha_param.data}")
        # exit(0)

        x = [x[:,i:i+1,:,:,:] for i in range(self.in_chans)]
        x = self.encoder(x)

        if self.is_training and self.use_aug:
            aug_preds = []
            i = 0
            for modality_x in zip(*x):
                aug_preds.append(self.decoder_aug(modality_x))
                i+=1

        x, alignment_loss = self.transformer(x, mask)
        alignment_loss = torch.mean(torch.stack(alignment_loss))

        sep_preds = []
        for modality_x in x:
            sep_preds.append(self.decoder_sep(modality_x))

        fuse_preds = []
        for b in range(mask.shape[0]):
            modality_comb = ''.join(str(int(m)) for m in mask[b].tolist())
            selected_alpha = self.alpha[modality_comb]
            # print(f"Alpha {modality_comb}: grad={selected_alpha.grad}, value={selected_alpha.data}")
            alpha_soft = F.softmax(selected_alpha, dim=0).view(-1, self.num_classes, 1, 1, 1)
            sep_preds_b = [sep_preds[k][b] for k in range(mask.shape[1])]
            fuse_pred_b = torch.zeros_like(sep_preds_b[0])
            for l, sep_pred in enumerate(sep_preds_b):
                # print(f"alpha_soft {l}: grad={alpha_soft[l].grad}, value={alpha_soft[l].data}")
                fuse_pred_b += alpha_soft[l] * sep_pred.detach()
            fuse_preds.append(fuse_pred_b)
        fuse_pred = torch.stack(fuse_preds)
        
        # fuse_preds = []
        # for b in range(mask.shape[0]):
        #     modality_comb = ''.join(str(int(m)) for m in mask[b].tolist())
        #     selected_alpha = self.alpha[modality_comb]
        #     # print(f"Alpha {modality_comb}: grad={selected_alpha.grad}, value={selected_alpha.data}")
        #     valid_indices = torch.tensor([k for k in range(mask.shape[1]) if mask[b, k]], dtype=torch.long)
        #     selected_alpha_valid = selected_alpha[valid_indices]
        #     # print(f"selected_alpha_valid {valid_indices}: grad={selected_alpha_valid.grad}, value={selected_alpha_valid.data}")
        #     alpha_soft = F.softmax(selected_alpha_valid, dim=0).view(-1, self.num_classes, 1, 1, 1)
        #     sep_preds_b = [sep_preds[k][b] for k in valid_indices]
        #     fuse_pred_b = torch.zeros_like(sep_preds_b[0])
        #     for l, sep_pred in enumerate(sep_preds_b):
        #         # print(f"alpha_soft {l}: grad={alpha_soft[l].grad}, value={alpha_soft[l].data}")
        #         fuse_pred_b += alpha_soft[l] * sep_pred.detach()
        #     fuse_preds.append(fuse_pred_b)
        # fuse_pred = torch.stack(fuse_preds)

        if self.is_training:
            results = [fuse_pred, sep_preds]

            if self.use_aug:
                results.append(aug_preds)
            else:
                results.append(None)

            if any(self.missing_completion) and self.alignment_loss_type is not None:
                results.append(alignment_loss)
            else:
                results.append(None)

            return results

        return fuse_pred


if __name__ == "__main__":
    segmenter = Model(img_size=80, num_classes=4).cuda()
    input = torch.rand(1, 4, 80, 80, 80).cuda()
    mask = torch.rand(1, 4).cuda()
    pred = segmenter(input, mask)