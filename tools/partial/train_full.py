import time
import torch
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from dataset.shapenetv1 import Shapes
from models.I_RED_full import I_RED_full

torch.autograd.set_detect_anomaly(False)
import argparse
from options.base_options import BaseOptions, combine_flags
from options.full_options import FullOptions
from options.dataset_options import ShapeNetV1_options
from losses.full_loss import Full_loss
from visualization.vis_pc import show_pc


def train():
    arg_parser = argparse.ArgumentParser()
    base_options = BaseOptions()
    full_options = FullOptions()
    dataset_options = ShapeNetV1_options()
    ### add option
    arg_parser = base_options.initialize(arg_parser)
    arg_parser = full_options.initialize(arg_parser)
    opt = arg_parser.parse_args()
    ### load config
    opt = combine_flags(opt, config_path=opt.config)
    ### about TensorBoard
    writer = SummaryWriter(logdir=opt.log_dir)
    ### load dataconfig
    opt_dataset = dataset_options.combine_configs(file_path=opt.config)
    ### create dataset
    dataset = Shapes(opt_dataset)
    ### len of dataset,img
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.full_batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=dataset.collate,
                                                   num_workers=2,
                                                   worker_init_fn=(lambda id:
                                                                   np.random.seed(np.random.get_state()[1][0] + id)))
    model = I_RED_full(opt).to(opt.base_device)
    if opt.full_resume:
        model.load_state_dict(torch.load(opt.full_resume_path))
        s_epoch = opt.full_resume_point
    else:
        s_epoch = 0

    ### init model train 
    total_iters = 0
    global_step = 0
    params = model.get_optim_params()
    optimizer = torch.optim.Adam(params, lr=opt.full_lr)
    lr_milestones = [opt.full_first_decay, opt.full_second_decay]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    ### init loss function
    Full_loss_func = Full_loss(opt)
    
    for epoch in range(s_epoch, opt.full_total_epochs):
        st_time = time.time()
        for i, data in enumerate(train_dataloader):
            total_iters += 1
            pc_full = data['target_shape']
            input_list = model.online_data_augment(pc_full)
            ####################
            '''
            show_pc(input_list['pc_full'][0].detach().cpu().numpy())
            show_pc(input_list['pc_part'][0].detach().cpu().numpy())
            show_pc(input_list['pc_fps'][0].detach().cpu().numpy())
            show_pc(input_list['pc_sample'][0].detach().cpu().numpy())
            '''

            ####################
            pred_recon, pred_add_noise, pred_fps, pred_remove_knn, \
            pred_resample, pred_randRT, pred_can_rot = model(input_list)
            loss_list, loss_sum = Full_loss_func.get_all_loss_terms(pred_recon, pred_add_noise,
                                                                    pred_fps, pred_remove_knn,
                                                                    pred_resample, pred_randRT,
                                                                    pred_can_rot)
            ####################
            '''
            show_pc(pred_recon['pc'][0].detach().cpu().numpy())
            show_pc(pred_recon['pc_at_inv'][0].detach().cpu().numpy())
            show_pc(pred_recon['recon_pc'][0].detach().cpu().numpy())
            show_pc(pred_recon['recon_pc_inv'][0].detach().cpu().numpy())
            '''
            ####################
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()

            if total_iters % opt.full_log_intervals == 0:
                writer.add_scalar(tag='sum_loss', scalar_value=loss_sum, global_step=global_step)
                writer.add_scalar(tag='recon_loss_1', scalar_value=loss_list['recon_1'], global_step=global_step)
                writer.add_scalar(tag='recon_loss_2', scalar_value=loss_list['recon_2'], global_step=global_step)
                writer.add_scalar(tag='ortho', scalar_value=loss_list['ortho'], global_step=global_step)
                #writer.add_scalar(tag='can_rot', scalar_value=loss_list['can_rot'], global_step=global_step)
                #writer.add_scalar(tag='can_T', scalar_value=loss_list['can_T'], global_step=global_step)
                #writer.add_scalar(tag='noised_rot', scalar_value=loss_list['noised_rot'], global_step=global_step)
                #writer.add_scalar(tag='noised_T', scalar_value=loss_list['noised_T'], global_step=global_step)
                #writer.add_scalar(tag='part_rot', scalar_value=loss_list['knn_rot'], global_step=global_step)
                #writer.add_scalar(tag='part_T', scalar_value=loss_list['knn_T'], global_step=global_step)
                #writer.add_scalar(tag='sample_rot', scalar_value=loss_list['resample_rot'], global_step=global_step)
                #writer.add_scalar(tag='sample_T', scalar_value=loss_list['resample_T'], global_step=global_step)
                #writer.add_scalar(tag='fps_rot', scalar_value=loss_list['fps_rot'], global_step=global_step)
                #writer.add_scalar(tag='fps_T', scalar_value=loss_list['fps_T'], global_step=global_step)
                #writer.add_scalar(tag='randRT', scalar_value=loss_list['randrt'], global_step=global_step)
                global_step += 1
                #print('Training loss sum: {0:04f}, global step: {1:03d}'.format(loss_sum, global_step))
                print('Training loss recon_loss_1: {0:04f}, recon_loss_2: {1:04f}, ortho: {2:04f}, sum: {3:04f}, global step: {4:03d}'.\
                    format(loss_list['recon_1'], loss_list['recon_2'], loss_list['ortho'], loss_sum, global_step))

        end_time = time.time()
        # update according to the epoch
        scheduler.step()
        if (epoch + 1) % opt.full_save_intervals == 0 or (epoch + 1) == opt.full_total_epochs:
            torch.save(model.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.log_dir, epoch))
        print('Training Epoch: {0} Finished, using {1:04f}.'.format(epoch, end_time - st_time))


if __name__ == '__main__':
    train()
