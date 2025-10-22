import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import random
import lpips

import warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights'", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

from skimage.metrics import structural_similarity

def setup_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mse_fn(pred, gt):
    return ((pred - gt)**2).mean()

def psnr_fn(pred, gt):
    return -10. * torch.log10(mse_fn(pred, gt))

def ssim_fn(pred, gt):
    '''
    input shape
        image: (W, H, C)
        video: (B, W*H, C)  and B is almost 1.
    '''
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    if pred.shape[2] == 3:
        ssims = []
        diffs = []
        for i in range(3):
            try:
                # for image
                score, diff = structural_similarity(pred[:,:,i], gt[:,:,i], full=True, data_range=1.)
            except ValueError:
                # for video
                score, diff = structural_similarity(pred[0, :,i], gt[0,:,i], full=True, data_range=1.)
            ssims.append(score)
            diffs.append(diff)
        #return np.array(ssims).mean(), np.array(diffs).mean()
        return np.array(ssims).mean()
    else:
        # gray image
        score, _ = structural_similarity(np.squeeze(pred), np.squeeze(gt), full=True, data_range=1.)
        return score
        #return score, _
        
calcluate_LPIPS = lpips.LPIPS(net='alex', verbose=False).cuda()
def lpips_fn(pred, gt):
    # input shape must be [H, W, C]
    return calcluate_LPIPS(pred.permute(2,0,1), gt.permute(2,0,1)).squeeze().cpu().detach().numpy()

def read_image(im_path, is_gray=False):
    if is_gray:
        im = imageio.imread(im_path, mode='L')  # mode F: 실수 [0, 1] 범위
        im = np.array(im).astype(np.float32) / 255.
        im = np.expand_dims(im, axis=-1)  # C 차원을 추가하여 (1, H, W)로 만듦
    else:
        im = imageio.imread(im_path, pilmode='RGB')
        im = np.array(im).astype(np.float32) / 255.
    return im

def write_image(im, im_path):
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(im_path, im)