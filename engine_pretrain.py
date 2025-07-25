# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, bool_masked_pos = batch

        samples = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # 环境依赖
        import numpy as np
        import cv2
        import os
        temp_path = args.log_dir + '/vis/'

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        if args.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = model(samples, mask=bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():
                # loss, _, _ = model(samples, mask=bool_masked_pos)
                ## 增加可视化操作：
                loss, imgs, pred = model(samples, mask=bool_masked_pos,vis=True)
                ## 选取部分通道可视化
                img_rgb =  (np.transpose(imgs[0, [7,13,19], :, :].cpu().numpy(),(1, 2, 0))* 255).astype(np.uint8)
                pred_rgb = (np.transpose(pred.squeeze(1)[0, [7,13,19], :, :].cpu().detach().numpy(),(1, 2, 0))* 255).astype(np.uint8)
                img_rgb = np.sqrt(img_rgb / 255.0) * 255.0
                pred_rgb = np.sqrt(pred_rgb / 255.0) * 255.0
                # cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_image_1.png'.format(epoch,data_iter_step), img[:,:,1].astype(np.uint8))
                # cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_pred_1.png'.format(epoch,data_iter_step), pred[:,:,1].astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_image.png'.format(epoch,data_iter_step), img_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_pred.png'.format(epoch,data_iter_step), pred_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ 'output_image.png'.format(epoch,data_iter_step), img_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ 'output_pred.png'.format(epoch,data_iter_step), pred_rgb.astype(np.uint8))
                img_mean = imgs[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1]).mean(dim=-1).cpu().numpy()
                img_sample = imgs[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1])[:,int(img_rgb.shape[0] * img_rgb.shape[1]/2)].cpu().numpy()
                pred_mean = pred[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1]).mean(dim=-1).cpu().detach().numpy()
                pred_sample = pred[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1])[:,int(img_rgb.shape[0] * img_rgb.shape[1]/2)].cpu().detach().numpy()
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
                plt.figure(figsize=(10, 6))
                plt.plot(img_mean, label='img_mean', linewidth=2, linestyle='-', color='C0')
                plt.plot(img_sample, label='img_sample', linewidth=2, linestyle='-', color='C1')
                plt.plot(pred_mean, label='pred_mean', linewidth=2, linestyle='--', color='C0')
                plt.plot(pred_sample, label='pred_sample', linewidth=2, linestyle='--', color='C1')
                plt.title('Comparison of Four Similar Curves')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                save_path = args.log_dir + '/vis/'+ '{}_{}_output_plt.png'.format(epoch,data_iter_step)
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.savefig(args.log_dir + '/vis/'+ 'output_plt.png'.format(epoch,data_iter_step), dpi=100, bbox_inches='tight')
                plt.close()
                
                


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}