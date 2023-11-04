import os
from utils.datasets import create_dataloader
import torch
import numpy as np
import cv2

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main():
    hyp='./data/hyps/hyp.scratch.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            import yaml
            hyp = yaml.safe_load(f)  # load hyps dict
    train_loader, train_dataset = create_dataloader('./data/cityscapes', is_train=True, img_size=640, batch_size=16 // WORLD_SIZE,
                                                    stride=32, hyp=hyp, augment=True, cache=False)
    #img, label = train_dataset[0]

    it = iter(train_loader)
    imgs, targets = next(it)


    if isinstance(imgs, torch.Tensor):
        img = imgs[0].cpu().float().numpy()   # img: (C, H, W)
    if isinstance(targets, torch.Tensor):
        label = targets[0].cpu().numpy()       # label: (H, W) 
    if np.max(img) <= 1:
        img *= 255  # de-normalize (optional)
    img = img.transpose(1,2,0).astype(np.uint8)  # (C, H, W) -> (H, W, C)    
    h, w, _ = img.shape
    cv2.imwrite('test_x.png', img)
    cv2.imwrite('test_y.png', label)
    return 0



if __name__ == "__main__":
    main()    
    



