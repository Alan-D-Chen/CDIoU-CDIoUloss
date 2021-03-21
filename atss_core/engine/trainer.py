# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import torch
from atss_core.config import cfg
import torch.distributed as dist

from atss_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from atss_core.utils.metric_logger import MetricLogger
stamps = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    ## logging ##
    logger = logging.getLogger("atss_core.trainer")
    logger.info("\n^_^-->The Program is Starting training:\n")
    meters = MetricLogger(delimiter="  ")
    infox = MetricLogger(delimiter="  ")

    ## SETTING FOR "maximum number of iterations" ##
    max_iter = len(data_loader)
    print("max_iter:----->", max_iter)
    print("max_iter:----->", max_iter)
    # max_iter = 20000
    # max_iter = cfg.SOLVER.EPOCH * len(data_loader)
    # max_iter = cfg.SOLVER.MAX_ITER
    """
    checkpointers = torch.load(cfg.MODEL.LOAD_PTH)
    model = torch.load(cfg.MODEL.LOAD_PTH)
    model = torch.load(cfg.MODEL.LOAD_PTH)
    
    print("checkpointers.keys():\n", checkpointers.keys())
    print("type of model:\n", type(checkpointers["model"]))
    print("checkpointers[model].keys():\n", checkpointers["model"].keys())
    print("type of iteration:\n", type(checkpointers["iteration"]))
    print("checkpointers[iteration].keys():\n", checkpointers["iteration"].keys())
    model.load(checkpointers["model"])
    model.load_state_dict(torch.load(cfg.MODEL.LOAD_PTH))
    """

    ## To load trained model .pth ##
    #PATH = "/home/alanc/Documents/ATSS/training_dir/atss_R_50_FPN_1x-3/model_final_2020-11-11_20-22-08.pth"
    ##################################################################################
    #print("The model is :\n",model)
    #print("The state dict of model:\n",model.state_dict)
    # print("The state dict of model:\n")
    # for parameters in model.parameters():
    #     print(parameters)
    ####################################################################################
    #PATH = "/home/alanc/Documents/ATSS/trained_pth/ATSS_R_101_FPN_2x.pth"
    #PATH = "/home/alanc/Documents/ATSS/trained_pth2/ATSS_dcnv2_R_101_FPN_2x.pth"
    #PATH = "/home/alanc/Documents/ATSS/trained_pth2/ATSS_X_101_32x8d_FPN_2x.pth"
    #PATH = "/home/alanc/Documents/ATSS/trained_pth2/ATSS_dcnv2_X_101_32x8d_FPN_2x.pth"
    PATH = "/home/alanc/Documents/ATSS/trained_pth/ATSS_dcnv2_X_101_64x4d_FPN_2x.pth"
    # model.load_state_dict(torch.load(PATH)["model"], strict=False)

    # model.load_state_dict(torch.load(cfg.MODEL.LOAD_PTH)["model"], strict=False)

    # Checkpoint = torch.load(PATH)
    # model_dict = model.state_dict()
    # model_dict.update(Checkpoint)
    # model.load_state_dict(model_dict, strict=False)
    pretrained_dict = torch.load(PATH)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    ###################################################################################
    #print("The new model is :\n", model)
    #print("The state dict of new model:\n", model.state_dict)
    # print("The state dict of new model:\n")
    # for parameters in model.parameters():
    #     print(parameters)
    ####################################################################################
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    ## JUST FOR TRAINING ##
    for j in range(1, cfg.SOLVER.EPOCH + 1, 1):
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                scheduler.step()

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if pytorch_1_1_0_or_later:
                scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            infox.update(time=batch_time, data=data_time)

            eta_seconds = infox.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            """
            Form of output
            """
            if iteration % 50 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "EPOCH: {EPOCH}",
                            "eta: {eta}",
                            "iter: {iter}",
                            "\n->{meters}",
                            "\n->Running info: {info}",
                            "\n->Learning Rate(lr): {lr:.6f}",
                            "\n->Max mem: {memory:.0f}",
                        ]
                    ).format(
                        EPOCH=str(j),
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        info=str(infox),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{}_{:07d}_{}".format(j, iteration, stamps), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_{}_{:07d}_{}".format(j, iteration, stamps), **arguments)
                print("^_^-->The program has reached the maximum number of iterations(max_iter) and has been stopped")
                break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
