import os
import yaml
import torch
import logging
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from data import YoloLoader
from option.option import Option
from utils.norm_trainer import Trainer
from utils.warmup import warmup_learning_rate
from utils.general import increment_path, set_logging, setup_seed

logger = logging.getLogger(__name__)

def main(local_rank, opt):
    setup_seed(2)
    set_logging(local_rank)
    # Show GPUs info
    if local_rank in [-1, opt.gpu_ids[0]]:
        s = f"Yolov3 detection training factory torch {torch.__version__}"
        space = " " * len(s)
        for i, d in enumerate(opt.gpu_ids):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 **2}MB)\n"
        logger.info(s)
    # Make Diectories
    wdir = opt.save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    save_model = wdir / "last.pth"
    opt.results_file = str(opt.save_dir / "results.txt")

    # Save run setings
    with open(opt.save_dir /"opt.yaml", "w") as f:
        opt.save_dir = str(opt.save_dir)
        yaml.dump(vars(opt), f, sort_keys=False)

    # Build Trainer
    trainer = Trainer(opt)
    # Data Loader
    data_loader = YoloLoader(opt)
    if opt.dist:
        # DDP mode
        torch.distributed.init_process_group(backend="nccl", rank=local_rank, world_size=opt.world_size)
        torch.cuda.set_device(local_rank)
        train_set = data_loader.data_set
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(train_set, opt.batch_size, shuffle=False,
                                                   sampler=sampler_train, num_workers=opt.load_thread,
                                                   collate_fn=train_set.collate_fn, pin_memory=True)
        trainer.model.cuda(local_rank)
        # SyncBatchNorm
        if local_rank != -1:
            process_group = torch.distributed.new_group()
            trainer.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model, process_group)
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model, device_ids=[local_rank], 
                                                                  output_device=local_rank, find_unused_parameters=True)
    else:
        # DP mode
        train_loader = data_loader.GetDataset()
        if len(opt.gpu_ids) > 1:
            trainer.model = torch.nn.DataParallel(trainer.model.cuda(device=opt.gpu_ids[0]), device_ids=opt.gpu_ids)
        else:
            trainer.model = trainer.model.cuda(device=opt.gpu_ids[0])
    logger.info(f"Init model complete.")
    # Train
    cudnn.benchmark = True
    ns = len(train_loader)
    loss_list =[]
    logger.info(f"Using {train_loader.num_workers} dataloader workers \n"
                f"Logging results to {opt.save_dir}\n"
                f"Starting training for {opt.max_epoch - trainer.opt.start_epoch} epochs..." )
    # Training process
    for epoch in range(trainer.opt.start_epoch, opt.max_epoch):
        mloss = torch.zeros(1, device=opt.gpu_ids[0] if local_rank==-1 else local_rank)
        xy_mloss = torch.zeros(1, device=opt.gpu_ids[0] if local_rank==-1 else local_rank)
        wh_mloss = torch.zeros(1, device=opt.gpu_ids[0] if local_rank==-1 else local_rank)
        obj_mloss = torch.zeros(1, device=opt.gpu_ids[0] if local_rank==-1 else local_rank)
        cls_mloss = torch.zeros(1, device=opt.gpu_ids[0] if local_rank==-1 else local_rank)

        if local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        trainer.model.train()
        pbar = enumerate(train_loader)
        logger.info(("%10s" * 8) % ("Epoch", "gpu_mem", "xy", "wh", "obj", "cls", "loss", "lr"))
        if local_rank in [-1, opt.gpu_ids[0]]:
            pbar = tqdm(pbar, total=ns, desc="Training")
        trainer.optim.zero_grad()
        # Batch process
        for i, data in pbar:
            # Forward and Backward
            loss = trainer.process(data)
            # Warm up
            if opt.warm_up and epoch < opt.warm_up_epoch:
                warmup_learning_rate(opt, trainer.optim, (i + 1)*(epoch + 1) / ns)
            # Display
            if local_rank in [-1, opt.gpu_ids[0]]:
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                xy_mloss = (xy_mloss * i + loss[0].item()) / (i + 1)
                wh_mloss = (wh_mloss * i + loss[1].item()) / (i + 1)
                obj_mloss = (obj_mloss * i + loss[2].item()) / (i + 1)
                cls_mloss = (cls_mloss * i + loss[3].item()) / (i + 1)
                mloss = (mloss * i + loss[4].item()) / (i + 1)
                s = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch + 1, opt.max_epoch), mem, *xy_mloss, *wh_mloss,
                                                   *obj_mloss, *cls_mloss, *mloss, trainer.optim.param_groups[0]["lr"])
                pbar.set_description(s)
            # end batch ----------------------------------------------
        # end epoch --------------------------------------------------
        # Scheduler
        if not (opt.warm_up and epoch < opt.warm_up_epoch):
            trainer.scheduler.step()
        
        if local_rank in [-1, opt.gpu_ids[0]]:
            # Write 
            with open(opt.results_file, "a") as f:
                f.write(s + "\n")
            # Plot loss
            loss_list.append(mloss)
            # Save model
            with open(opt.results_file, "r") as f:
                ckpt = trainer.model.state_dict()
            # Save last and delete
            if (epoch + 1) % opt.save_epoch == 0:
                torch.save(ckpt, save_model)
            torch.save(ckpt, save_model)
            del ckpt

if __name__=="__main__":
    op = Option()
    opt = op.parse()
    opt.start_epoch = 0
    opt.save_dir = increment_path(Path(opt.project) / opt.name, sep="_")
    # Choose DDP mode or DP mode
    if opt.dist:
        # DDP mode
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "8010"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids).replace('[', '').replace(']','')
        assert torch.cuda.device_count() > opt.local_rank
        torch.multiprocessing.spawn(main, args=(opt,), nprocs=opt.world_size, join=True)
    else:
        # DP mode
        main(local_rank=-1, opt=opt)