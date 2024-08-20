from vary.model.vision_encoder.sam import build_sam_vit_b


if __name__ == "__main__":
    vision_tower_high = build_sam_vit_b(checkpoint='/mnt/ceph2/pretrained/facebook/sam_vit_b_01ec64.pth')
    print(vision_tower_high)
