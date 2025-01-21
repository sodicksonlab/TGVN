import numpy as np
import torch, argparse, fastmri

from distributed import init_distributed_mode
from torch import optim, nn
from fastmri.data.subsample import RandomMaskFunc, EquiSpacedMaskFunc
from fastmri.data.transforms import center_crop_to_smallest
from data import VarNetDataTransformM4, SliceDatasetM4
from models import VarNetImage, VarNetImageSDC_2S
from custom_losses import MS_SSIM_L1Loss
from fastmri import SSIMLoss

def get_arguments():
    parser = argparse.ArgumentParser(description="Multi-coil VarNet", add_help=False)

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
    parser.add_argument("--type", type=str, default='std' )

    return parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multi-coil VarNet', parents=[get_arguments()])
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
    train_transform = VarNetDataTransformM4(
        flair_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
        use_seed=False
    )
    
    train_dataset = SliceDatasetM4(
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
    val_transform = VarNetDataTransformM4(
        flair_mask_func=EquiSpacedMaskFunc(center_fractions=[args.center_freq], accelerations=[args.acc]),
        use_seed=True
    )
    val_dataset = SliceDatasetM4(
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
    
    if args.type.lower() == 'std':
        print('————Standard VarNet, no secondary data consistency————')
        model = VarNetImage(num_cascades=args.num_casc, chans=args.num_chans).to(gpu)
    elif args.type.lower() == 'sdc':
        print('————Secondary data consistency————')
        model = VarNetImageSDC_2S(num_cascades=args.num_casc, chans=args.num_chans).to(gpu)
    else:
        raise NotImplementedError('There is no such type, check the arguments!')
        
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
            t2_kspace = batch.t2_kspace.cuda(gpu, non_blocking=True)
            flair_masked_kspace = batch.flair_kspace.cuda(gpu, non_blocking=True)
            flair_mask = batch.flair_mask.cuda(gpu, non_blocking=True)
            flair_target = batch.flair_target.cuda(gpu, non_blocking=True)
            flair_mx = batch.flair_max_value.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            
            # At this point out is of size BxHxW (magnitude-image)
            if args.type.lower() == 'std':
                out = model(flair_masked_kspace, flair_mask)
            else:
                out = model(flair_masked_kspace, flair_mask, t2_kspace)
            target, out = center_crop_to_smallest(flair_target.unsqueeze(1), out)
            l = loss(out, target, data_range=flair_mx.item())
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
            torch.save(state, f'./TGVN_{args.acc}x_{args.type}_{epoch}.pth')
        
        with torch.no_grad():
            model.eval()
            val_sampler.set_epoch(epoch)
            for step, batch in enumerate(val_loader, start=epoch*len(val_loader)):
                t2_kspace = batch.t2_kspace.cuda(gpu, non_blocking=True)
                flair_masked_kspace = batch.flair_kspace.cuda(gpu, non_blocking=True)
                flair_mask = batch.flair_mask.cuda(gpu, non_blocking=True)
                flair_target = batch.flair_target.cuda(gpu, non_blocking=True)
                flair_mx = batch.flair_max_value.cuda(gpu, non_blocking=True)
                
                if args.type.lower() == 'std':           
                    out = model(flair_masked_kspace, flair_mask)
                else:
                    out = model(flair_masked_kspace, flair_mask, t2_kspace)
                    
                target, out = center_crop_to_smallest(flair_target.unsqueeze(1), out)
                s = 1 - ssim_loss(out, target, data_range=flair_mx) 
                l = loss(out, target, data_range=flair_mx.item())
                val_loss += l.item()
                val_ssim += s.item()

            val_loss /= len(val_loader)
            val_ssim /= len(val_loader)

        print(f'val_loss at epoch {epoch} is {val_loss}, val_ssim is {val_ssim}')
        scheduler.step()
