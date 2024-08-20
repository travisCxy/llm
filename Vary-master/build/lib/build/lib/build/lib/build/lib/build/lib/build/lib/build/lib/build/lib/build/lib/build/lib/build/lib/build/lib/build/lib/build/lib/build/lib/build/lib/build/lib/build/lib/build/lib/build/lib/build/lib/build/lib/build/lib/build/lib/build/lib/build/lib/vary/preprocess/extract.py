import torch
from vary.model import *


if __name__ == '__main__':
    model = varyOPTForCausalLM.from_pretrained('/mnt/ceph2/Vary/runs/0518/checkpoint-17000')
    vision_tower_high = model.get_model().vision_tower_2
    # for k, v in vision_tower_high.state_dict().items():
    #     if 'vision_tower_high' in k:
    #         print(k, v.shape)
    name = 'patch_embed.proj.bias'
    print(vision_tower_high.state_dict()[name][:10])
    torch.save(vision_tower_high.state_dict(), '/mnt/ceph2/Vary/runs/0518/vision_tower_high.pth')
