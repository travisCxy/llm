import torch
from vary.model import *
from vary.model.vision_encoder.sam import build_sam_vit_b, build_sam_vit_l


if __name__ == '__main__':
    # model = varyOPTForCausalLM.from_pretrained('/mnt/ceph2/Vary/runs/05212/')
    # vision_tower_high = model.get_model().vision_tower
    # # for k, v in vision_tower_high.state_dict().items():
    # #     if 'vision_tower_high' in k:
    # #         print(k, v.shape)
    # name = 'patch_embed.proj.bias'
    # print(vision_tower_high.state_dict()[name][:10])
    # torch.save(vision_tower_high.state_dict(), '/mnt/ceph2/Vary/runs/05212/vision_tower_high.pth')
    vision_tower_high = build_sam_vit_l(checkpoint='/mnt/ceph2/Vary/runs/05212/vision_tower_high_1280.pth')
    with torch.set_grad_enabled(False):
        cnn_feature = vision_tower_high(torch.randn(1, 3, 1280, 1280))
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # 256*1024
    print(cnn_feature.shape)

    # import torch.nn.functional as F
    # state_dict = torch.load('/mnt/ceph2/Vary/runs/05212/vision_tower_high.pth', map_location='cpu')
    # image_size = 1024
    # new_image_size = 1280
    # vit_patch_size = 16
    # if 'pos_embed' in state_dict:
    #     pos_embed = state_dict['pos_embed']
    #     old_size = pos_embed.shape[1]
    #     new_size = new_image_size // vit_patch_size
    #     pos_embed = pos_embed.permute(0, 3, 1, 2)  # 将(1, 64, 64, 1024)调整为(1, 1024, 64, 64)
    #     interpolated_pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bicubic')
    #     interpolated_pos_embed = interpolated_pos_embed.permute(0, 2, 3, 1)  # 将(1, 1024, 80, 80)调整为(1, 80, 80, 1024)
    #     state_dict['pos_embed'] = interpolated_pos_embed
    # encoder_depth = 24
    # for i in range(encoder_depth):
    #     if f'blocks.{i}.attn.rel_pos_h' in state_dict:
    #         rel_pos_h = state_dict[f'blocks.{i}.attn.rel_pos_h']
    #         rel_pos_w = state_dict[f'blocks.{i}.attn.rel_pos_w']
    #         old_size = rel_pos_h.shape[0]
    #         if old_size != 2 * (image_size // vit_patch_size) - 1:
    #             continue
    #         print(f'blocks.{i}.attn.rel_pos_h')
    #         new_size = 2 * (new_image_size // vit_patch_size) - 1
    #         interpolated_rel_pos_h = F.interpolate(rel_pos_h.unsqueeze(0).unsqueeze(0), size=(new_size, rel_pos_h.shape[1]), mode='bicubic')
    #         interpolated_rel_pos_w = F.interpolate(rel_pos_w.unsqueeze(0).unsqueeze(0), size=(new_size, rel_pos_w.shape[1]), mode='bicubic')
    #         state_dict[f'blocks.{i}.attn.rel_pos_h'] = interpolated_rel_pos_h.squeeze(0).squeeze(0)
    #         state_dict[f'blocks.{i}.attn.rel_pos_w'] = interpolated_rel_pos_w.squeeze(0).squeeze(0)
    # torch.save(state_dict, '/mnt/ceph2/Vary/runs/05212/vision_tower_high_1280.pth')
