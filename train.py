# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import torch
from torchvision.utils import save_image
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from util.util import mkdir
from trainers.pix2pix_trainer import Pix2PixTrainer
from torch.utils.data import DataLoader, ConcatDataset
# from data.PSF_dataset_sample import PSFDataset
# from data.tif_dataset import TifDataset
# from data.wfsless_dataset import WFSDataset
from data.SH_dataset import SHDataset
from torch.utils.data import DataLoader
import numpy as np

if __name__ == '__main__':
    
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))
    # dataloader = data.create_dataloader(opt)` 

    # data_path = "/disk/sdb2/PutYourFilesHere/lc/far_field_data"
    # h5_folder = "/disk/sdb2/PutYourFilesHere/lc/far_field_data/h5_data"
    # mat_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/phase_maps'
    # train_dataset = PSFDataset(opt, data_path, h5_folder, mat_folder,
    #                        train=True, split=0.9, num_samples=15000,
    #                        seed=42, index_file='train_indices_0318.npy')

    # data_path = "/disk/sdb2/PutYourFilesHere/lc/far_field_data"
    # tif_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/0318_data/tif_data_1'
    # mat_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/0318_data/phase_maps_1'
    # train_dataset = TifDataset(opt, data_path, tif_folder, mat_folder,
    #                      train=True, split=0.95, num_samples=4900, seed=52, 
    #                      index_file='train_indices_0319.npy')

    data_path = opt.dataroot
    train_dataset = SHDataset(opt, data_path, train=True, split=0.9)

    train_dataset.initialize(opt)
    dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=(opt.phase == 'train'),
        num_workers=int(opt.nThreads),
        drop_last=(opt.phase == 'train'),
    )
    len_dataloader = len(dataloader)
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))
    # create trainer for our model
    trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)
    save_root = os.path.join('checkpoints', opt.name, 'train')
    mkdir(save_root)

    for epoch in iter_counter.training_epochs():
        trainer.set_epoch(epoch)
        
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)
            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                try:
                    print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                         iter_counter.epoch_iter_num, losses, iter_counter.time_per_iter)
                except OSError as err:
                    print(err)

            if iter_counter.needs_displaying():
                imgs_num = data_i['label'].shape[0]

                if opt.dataset_mode == 'SH':
                    label = data_i['label'][:,:3,:,:]

                show_size = opt.display_winsize

                output=trainer.get_latest_generated().data.cpu()
                output = output * 50
                output = ((output - torch.min(output)) / (torch.max(output) - torch.min(output)))
                image = data_i['image'].cpu() * 50
                image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                imgs = torch.cat((label.cpu(),  \
                                output, \
                                image), 0)
                if torch.isnan(imgs).any():
                    print('nan')

                try:
                    save_name = '%04d_%04d.png' % (epoch, iter_counter.total_steps_so_far)
                    save_name = os.path.join(save_root, save_name)
                    save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
                except OSError as err:
                    print(err)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                try:
                    trainer.save('latest')
                    iter_counter.record_current_iter()
                except OSError as err:
                    import pdb; pdb.set_trace()
                    print(err)

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                trainer.save(epoch)
            except OSError as err:
                print(err)

    print('Training was successfully finished.')
