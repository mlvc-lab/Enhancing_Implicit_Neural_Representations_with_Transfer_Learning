import os
import time
import torch
import numpy as np
import imageio.v2 as imageio
from tqdm import trange
from utils import read_image, write_image, setup_seed, ssim_fn, lpips_fn
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose
from models import *
import configargparse
import json
import copy



tonp = lambda x: x.cpu().detach().numpy()
mse_fn = lambda pred, gt: ((pred - gt)**2).mean()

def psnr_fn(pred, gt):
    mse = mse_fn(pred.clip(0, 1), gt)
    if isinstance(pred, torch.Tensor):
        return -10 * torch.log10(mse)
    return -10 * np.log10(mse)


def get_opts():
    parser = configargparse.ArgumentParser()
    # test
    parser.add_argument('--test', action='store_true')
    
    # data
    parser.add_argument('--imgid', type=int, default=1)
    parser.add_argument('--datadir', type=str, default='data/div2k/test_data/')
    parser.add_argument('--specific_img', type=str, default=None, help="If not None, use specific image path instead of imgid")
    
    # model
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--out_features', type=int, default=3)  
    parser.add_argument('--hidden_layers', type=int, default=3) 
    parser.add_argument('--hidden_features', type=int, default=256)
    # 
    parser.add_argument('--model_type', type=str, default='Finer', required=['Finer', 'Siren', 'Wire', 'Gauss', 'GF', 'WF'])
    parser.add_argument('--first_omega', type=float, default=30)
    parser.add_argument('--hidden_omega', type=float, default=30)
    parser.add_argument('--omega', type=float, default=5)   
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--omega_w', type=float, default=20)
    parser.add_argument('--N_freqs', type=int, default=10)
    # 
    parser.add_argument('--fbs', type=float, default=None,  help='bias_scale of the first layer')
    parser.add_argument('--hbs', type=float, default=None)
    parser.add_argument('--init_method', type=str, default='pytorch', required=['sine', 'pytorch'])
    parser.add_argument('--init_gain', type=float, default=1)
    ## Train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=5000) 
    parser.add_argument('--zero_mean', type=int, default=1)
    ## Log  
    parser.add_argument('--logdir', type=str, default='logs/Finer/')    
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--reuse', action='store_true') 
    parser.add_argument('--exp_suffix', type=str, default='') 
    ## My add param
    parser.add_argument('--is_gray_img', default = 0, type=int, help='0: RGB, 1: Gray')
    parser.add_argument('--side_len', type=int, default=256)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--pretrain_epoch', type=int, default=500)
    return parser.parse_args()


# Image Fitting 
def train_image(model, coords, gt, loss_fn=mse_fn, lr=5e-4, num_epochs=2000, steps_til_summary=1, invnorm=lambda x:x):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))
        
    train_iter = []
    train_psnr = []
    train_ssim = []
    train_lpips = []
    model_states = {}
    total_time = 0
    pred_imgs = {}
    best_psnr = 0.0
    for epoch in trange(1, num_epochs + 1):
        time_start = time.time()

        pred = model(coords)
        loss = loss_fn(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        total_time += time.time() - time_start
   
        if not epoch % steps_til_summary:
            with torch.no_grad():
                train_iter.append(epoch)
                pred = pred.reshape(size)
                psnr_curr = psnr_fn(invnorm(pred), invnorm(gt.reshape(size))).item()
                if psnr_curr > best_psnr:
                    best_psnr = psnr_curr
                    model_states['best'] = copy.deepcopy(model.state_dict())

                train_psnr.append(psnr_curr)
                train_ssim.append((ssim_fn(invnorm(pred), invnorm(gt.reshape(size)))).item())
                train_lpips.append((lpips_fn(invnorm(pred), invnorm(gt.reshape(size)))).item()) 
                
        ## For intermittent saving
        if epoch in [500, 2000]:
            model_states[str(epoch)] = copy.deepcopy(model.state_dict())

        if epoch in [1, 5, 10, 20, 50, 100, 2000]:
            pred_imgs[f'{epoch}step'] = invnorm(pred.reshape(size))

                
    # with torch.no_grad():
    #     pred = invnorm(model(coords))
    gt = invnorm(gt)
        
    ret_dict = {
        'train_iter': train_iter,
        'train_psnr': train_psnr,
        'train_ssim': train_ssim,
        'train_lpips': train_lpips,
        'pred': pred_imgs,
        'gt': gt,
        'model_state': model_states
    }
    return ret_dict


def imgid2path(imgid, datadir='/local_dataset/DIV2K'):
    path = None
    if datadir.find('DIV2K') != -1:
        path = os.path.join(datadir, '%02d.png'%(imgid))
    if datadir.find('Chest_CT') != -1:
        path = os.path.join(datadir, '%03d.png'%(imgid))
    if datadir.find('CelebA_HQ') != -1:
        path = os.path.join(datadir, '%02d.jpg'%(imgid))
    if path is None:
        raise ValueError('Unknown dataset')
    return path

def get_train_data(cfg):
    # data
    if not cfg.zero_mean:
        norm = lambda x : x
        invnorm = lambda x : x
    else:
        norm = lambda x : x*2-1
        invnorm = lambda x : x/2+0.5
        
    im_path = cfg.specific_img if cfg.specific_img is not None else imgid2path(cfg.imgid, cfg.datadir)
    
    im = read_image(im_path, cfg.is_gray_img)
    H, W = im.shape[:2]
    C = im.shape[2] if len(im.shape) == 3 else 1
    im = norm(im)
    
    aug_list = [
            ToTensor(),
            CenterCrop(min(H, W)),
            Resize((cfg.side_len, cfg.side_len)),
    ]

    transform = Compose(aug_list)
    img = transform(im).permute(1, 2, 0)
    H, W = img.shape[:2]
    C = img.shape[2] if len(img.shape) == 3 else 1
    # target
    im_gt = img.reshape(H*W, C)
    # input
    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'), dim=-1).reshape(-1, 2)    
    return coords, im_gt, invnorm, [H, W, C]    # pred는 항상 3채널로 진행


def get_model(cfg):
    mtype = cfg.model_type.lower()
    if mtype == 'finer':
        model = Finer(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                      first_omega=cfg.first_omega, hidden_omega=cfg.hidden_omega,
                      init_method=cfg.init_method, init_gain=cfg.init_gain, fbs=cfg.fbs)
    elif mtype == 'siren':
        model = Siren(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                      first_omega_0=cfg.first_omega, hidden_omega_0=cfg.hidden_omega)
    elif mtype == 'wire':
        model = Wire(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                     scale=cfg.scale, omega_w=cfg.omega_w,
                     init_method=cfg.init_method, init_gain=cfg.init_gain)
    elif mtype == 'wf':
        model = WF(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                   scale=cfg.scale, omega_w=cfg.omega_w, omega=cfg.omega,
                   init_method=cfg.init_method, init_gain=cfg.init_gain, fbs=cfg.fbs)
    elif mtype == 'gauss':
        model = Gauss(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                      scale=cfg.scale,
                      init_method=cfg.init_method, init_gain=cfg.init_gain)
    elif mtype == 'gf':
        model = GF(in_features=cfg.in_features, out_features=cfg.out_features, hidden_layers=cfg.hidden_layers, hidden_features=cfg.hidden_features,
                   scale=cfg.scale, omega=cfg.omega,
                   init_method=cfg.init_method, init_gain=cfg.init_gain, fbs=cfg.fbs)
    return model


def generate_expname(cfg):
    expname = \
        f"imid[{cfg.imgid}]_{cfg.model_type}_{cfg.hidden_layers}x{cfg.hidden_features}_" + \
        f"init[{cfg.init_method}]_fbs[{cfg.fbs}]_lr[{cfg.lr}]"
    #
    mtype = cfg.model_type.lower()
    if mtype == 'finer' or mtype == 'siren':    # first_omega, hidden_omega
        expname += f"_fw[{cfg.first_omega}]_hw[{cfg.hidden_omega}]"
    elif mtype == 'gauss' or mtype == 'gf':     # omega, scale
        expname += f"_omega[{cfg.omega}]_scale[{cfg.scale}]"
    elif mtype == 'wire' or mtype == 'wf':
        expname += f"_omega[{cfg.omega}]_scale[{cfg.scale}]_omegaw[{cfg.omega_w}]"
    elif mtype == 'pemlp':
        expname += f"_Nfreqs[{cfg.N_freqs}]"
    return expname + cfg.exp_suffix


if __name__ == '__main__':
    opts = get_opts()
        
    # print('--- Run Configuration ---')
    # for k, v in vars(opts).items():
    #     print(k, '=', v)
    # print('--- Run Configuration ---')
    
    setup_seed(0)
    
    # logdir 
    os.makedirs(opts.logdir, exist_ok=True)
    
    # expname
    expname = generate_expname(opts)
    
    # data
    coords, gt, invnorm, size = get_train_data(opts)
        
    if not opts.test:
        # model
        model = get_model(opts)
        
        if opts.load_path is not None:
            model.load_state_dict(torch.load(opts.load_path)['model_state'][f'{opts.pretrain_epoch}'])
        
        # to gpu
        device = torch.device('cuda:0')
        gt = gt.to(device)
        coords = coords.to(device)
        model = model.to(device)

        # train
        res = train_image(model, coords, gt, loss_fn=mse_fn, lr=opts.lr, num_epochs=opts.num_epochs, steps_til_summary=1, invnorm=invnorm)
        
        # save 
        torch.save(res, os.path.join(opts.logdir, f'{expname}.pt'))
        
    else:
        res = torch.load(os.path.join(opts.logdir, f'{expname}.pt'))
        pred = tonp(res['pred']).reshape(size)
        write_image(pred, os.path.join(opts.logdir, f'{expname}.png'))
        with open(os.path.join(opts.logdir, f'metrics.json'), 'w') as f:
            psnr = psnr_fn(pred, tonp(invnorm(gt)).reshape(size))
            json.dump({'PSNR': psnr}, f)
            print(f'PSNR: {psnr}')
        