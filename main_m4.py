import numpy as np
import torch, argparse, fastmri

from distributed import init_distributed_mode
from torch import optim, nn
from fastmri.data.subsample import RandomMaskFunc, EquiSpacedMaskFunc
from fastmri.data.transforms import center_crop_to_smallest
from data import VarNetDataTransformM4Joint, SliceDatasetM4Joint
from models import VarNetImage, TGVN_2S
from custom_losses import MS_SSIM_L1Loss
from fastmri import SSIMLoss

def get_arguments():
    parser = argparse.ArgumentParser(description="TGVN", add_help=False)

    # Distributed
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # VarNet params
    parser.add_argument("--num-casc", type=int, default=10)
    parser.add_argument("--num-chans", type=int, default=21)
    
    # Mask params
    parser.add_argument("--acc", type=int, default=8,
                        help='Acceleration factor')
    parser.add_argument("--center-freq", type=float, default=0.04)
    
    
    # Type and checkpoint location
    parser.add_argument("--ckpt-loc", type=str, default='none')
    parser.add_argument("--type", type=str, default='e2e')
    parser.add_argument("--main-contrast", type=str, default='flair')

    return parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser('TGVN', parents=[get_arguments()])
    args = parser.parse_args()
    torch.backends.cudnn.benchmark=True
    init_distributed_mode(args)
    print('Distributed mode initialized!')

    print(args)
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    challenge = 'multicoil'
    batch_size = torch.cuda.device_count()
    per_device_batch_size = 1 # collation can be a problem if > 1
    
    train_path = './m4raw_split/train.csv'
    if args.main_contrast.lower() == 'flair':
        train_transform = VarNetDataTransformM4Joint(
            flair_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
            t1_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            t2_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            use_seed=False
        )
    else:
        train_transform = VarNetDataTransformM4Joint(
            flair_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            t1_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
            t2_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            use_seed=False
        )        
    
    train_dataset = SliceDatasetM4Joint(
        csv_path=train_path,
        transform=train_transform,
        challenge=challenge
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=per_device_batch_size,
        num_workers=batch_size,
    )

    val_path = './m4raw_split/val.csv'
    if args.main_contrast.lower() == 'flair':
        val_transform = VarNetDataTransformM4Joint(
            flair_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
            t1_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            t2_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            use_seed=True
        )
    else:
        val_transform = VarNetDataTransformM4Joint(
            flair_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            t1_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
            t2_mask_func=EquiSpacedMaskFunc(center_fractions=[0], accelerations=[1]),
            use_seed=True
        )
        
    val_dataset = SliceDatasetM4Joint(
        csv_path=val_path,
        transform=val_transform,
        challenge=challenge
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=per_device_batch_size,
        num_workers=batch_size,
    )    
    
    if args.type.lower() == 'e2e':
        print('————End-to-end Variational Network, no side information————')
        model = VarNetImage(num_cascades=args.num_casc, chans=args.num_chans).to(gpu)
    elif args.type.lower() == 'tgvn':
        print('————Trust-guided Variational Network————')
        model = TGVN_2S(num_cascades=args.num_casc, chans=args.num_chans).to(gpu)
    else:
        raise NotImplementedError('There is no such type, check the arguments!')
        
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    if args.ckpt_loc.lower() != 'none':
        print('————Pretrained————')
        ckpt = torch.load(args.ckpt_loc)
        model.load_state_dict(ckpt['model'])
        print('Model loaded!')
    else:
        print('————Random Init————')

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of trainable parameters: {params}')
 
    num_epoch = 100
    lr_start = 3e-4
    gamma = 0.98

    print(f'num_epochs: {num_epoch}, lr_start: {lr_start}, gamma: {gamma}')
    loss = MS_SSIM_L1Loss(alpha=0.84, compensation=1.0).to(gpu)
    ssim_loss = SSIMLoss().to(gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
 
    print('Entering training loop')
    for epoch in range(num_epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0
        val_loss = 0
        val_ssim = 0
        for step, batch in enumerate(train_loader, start=epoch*len(train_loader)):
            t2_kspace = batch.t2_kspace.cuda(gpu, non_blocking=True)  # Side information
            flair_kspace = batch.flair_kspace.cuda(gpu, non_blocking=True)
            flair_mask = batch.flair_mask.cuda(gpu, non_blocking=True)
            flair_target = batch.flair_target.cuda(gpu, non_blocking=True)
            flair_mx = batch.flair_max_value.cuda(gpu, non_blocking=True)
            t1_kspace = batch.t1_kspace.cuda(gpu, non_blocking=True)
            t1_mask = batch.t1_mask.cuda(gpu, non_blocking=True)
            t1_target = batch.t1_target.cuda(gpu, non_blocking=True)
            t1_mx = batch.t1_max_value.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            
            # At this point out is of size BxHxW (magnitude-image)
            if args.type.lower() == 'e2e':
                if args.main_contrast.lower() == 'flair':
                    out = model(flair_kspace, flair_mask)
                else:
                    out = model(t1_kspace, t1_mask)
            else:
                if args.main_contrast.lower() == 'flair':
                    out = model(flair_kspace, flair_mask, t2_kspace)
                else:
                    out = model(t1_kspace, t1_mask, t2_kspace)
                    
            if args.main_contrast.lower() == 'flair':
                target, out = center_crop_to_smallest(flair_target.unsqueeze(1), out)
                l = loss(out, target, data_range=flair_mx.item())
            else:
                target, out = center_crop_to_smallest(t1_target.unsqueeze(1), out)
                l = loss(out, target, data_range=t1_mx.item())
                
            l.backward()
            optimizer.step()
            train_loss += l.item()

        train_loss /= len(train_loader)
        print(f'train_loss at epoch {epoch} is {train_loss}')
            
        if args.rank == 0:
            state = dict(
                epoch=epoch+1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, f'./model_{args.type}_{args.acc}x_{args.main_contrast}_{epoch}.pth')
        
        with torch.no_grad():
            model.eval()
            val_sampler.set_epoch(epoch)
            for step, batch in enumerate(val_loader, start=epoch*len(val_loader)):
                t2_kspace = batch.t2_kspace.cuda(gpu, non_blocking=True)
                flair_kspace = batch.flair_kspace.cuda(gpu, non_blocking=True)
                flair_mask = batch.flair_mask.cuda(gpu, non_blocking=True)
                flair_target = batch.flair_target.cuda(gpu, non_blocking=True)
                flair_mx = batch.flair_max_value.cuda(gpu, non_blocking=True)
                t1_kspace = batch.t1_kspace.cuda(gpu, non_blocking=True)
                t1_mask = batch.t1_mask.cuda(gpu, non_blocking=True)
                t1_target = batch.t1_target.cuda(gpu, non_blocking=True)
                t1_mx = batch.t1_max_value.cuda(gpu, non_blocking=True)
                
                if args.type.lower() == 'e2e':
                    if args.main_contrast.lower() == 'flair':
                        out = model(flair_kspace, flair_mask)
                    else:
                        out = model(t1_kspace, t1_mask)
                else:
                    if args.main_contrast.lower() == 'flair':
                        out = model(flair_kspace, flair_mask, t2_kspace)
                    else:
                        out = model(t1_kspace, t1_mask, t2_kspace)
                    
                if args.main_contrast.lower() == 'flair':
                    target, out = center_crop_to_smallest(flair_target.unsqueeze(1), out)
                    l = loss(out, target, data_range=flair_mx.item())
                    s = 1 - ssim_loss(out, target, data_range=flair_mx) 
                else:
                    target, out = center_crop_to_smallest(t1_target.unsqueeze(1), out)
                    l = loss(out, target, data_range=t1_mx.item())
                    s = 1 - ssim_loss(out, target, data_range=t1_mx) 
                    
                val_loss += l.item()
                val_ssim += s.item()

            val_loss /= len(val_loader)
            val_ssim /= len(val_loader)

        print(f'val_loss at epoch {epoch} is {val_loss}, val_ssim is {val_ssim}')
        scheduler.step()
