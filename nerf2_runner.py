# -*- coding: utf-8 -*-
"""NeRF2 runner for training and testing
"""

import os

from av import OutputChangedError

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import yaml
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.io as scio

from dataloader import *
from model import *
from renderer import renderer_dict
from utils.data_painter import paint_spectrum_compare
from utils.logger import logger_config
import nni
import math

def loss_function(recon_x, x, mu, logvar):
    # Binary Cross-Entropy Loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # Kullback-Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



class NeRF2_Runner():

    def __init__(self, mode, dataset_type, params, m, n, **kwargs) -> None:


        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['train']
        self.dataset_type = dataset_type
        n_samples = kwargs_render['n_samples']
        self.params_nni = params
        self.nni = n
        self.m = m
        self.experiment_id = nni.get_experiment_id()
        self.id = nni.get_trial_id()

        ## Path settings
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nerf2')
        self.logger.info("expname:%s, datadir:%s, logdir:%s", self.expname, self.datadir, self.logdir)
        self.writer = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))


        ## Networks
        self.nerf2_network = NeRF2(n_samples=n_samples, m = self.m,  **kwargs_network).to(self.devices)
        if self.dataset_type == 'ble':
            self.MLP = MLP(input_size=978, hidden_size1=512, hidden_size2=216, hidden_size3=128, output_size=1).to(self.devices)
            self.Unet = UNet1D(in_channels=978, base_channels=64, out_channels=1).to(self.devices)
            self.VAE = VAE(input_dim=978, hidden_dim=512, latent_dim=128).to(self.devices)
        elif self.dataset_type == 'mimo':
            self.MLP = MLP(input_size=1027, hidden_size1=512, hidden_size2=216, hidden_size3=128, output_size=1).to(self.devices)
            self.Unet = UNet1D(in_channels=1027, base_channels=64, out_channels=1).to(self.devices)
            self.VAE = VAE(input_dim=1027, hidden_dim=512, latent_dim=128).to(self.devices)
        if self.m == 'mlpvae':
            params = list(self.nerf2_network.parameters()) + list(self.coarse_network.parameters())
        else:
            params = list(self.nerf2_network.parameters())

        if self.nni == 'nni':
            self.optimizer = torch.optim.AdamW(params, lr=float(self.params_nni['lr']),
                                          weight_decay=float(self.params_nni['wd']),
                                          betas=(0.9, 0.999))
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                        T_0=kwargs_train['T_max'], eta_min=float(self.params_nni['em']), T_mult=2)
            # self.total_iterations = self.params_nni['epochs']
        else:
            self.optimizer = torch.optim.AdamW(params, lr=float(kwargs_train['lr']),
                                          weight_decay=float(kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                        T_0=5000, eta_min=float(kwargs_train['eta_min']),
                                                                        T_mult=2)
            # self.total_iterations = kwargs_train['total_iterations']
        
        self.total_iterations = kwargs_train['total_iterations']

        ## Renderer
        renderer = renderer_dict[kwargs_render['mode']]
        self.renderer = renderer(networks_fn=self.nerf2_network, m=self.m, **kwargs_render)
        # self.render_U_net = renderer(networks_fn=self.Unet, m=self.m, **kwargs_render)
        # self.render_VAE = renderer(networks_fn=self.VAE, m=self.m, **kwargs_render)
        self.scale_worldsize = kwargs_render['scale_worldsize']

        ## Print total number of parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        if kwargs_train['load_ckpt'] or mode == 'test':
            self.load_checkpoints()
        self.batch_size = kwargs_train['batch_size']
        self.save_freq = kwargs_train['save_freq']

        ## Dataset
        dataset = dataset_dict[dataset_type]
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type=dataset_type)
        self.logger.info("Loading training set...")
        self.train_set = dataset(self.datadir, train_index, self.scale_worldsize)
        self.logger.info("Loading test set...")
        self.test_set = dataset(self.datadir, test_index, self.scale_worldsize)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))


    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        trail_dir = os.path.join(ckptsdir, self.experiment_id, self.id)
        if not os.path.exists(trail_dir):
            os.makedirs(trail_dir)
        trail_dir = 'logs/NeWRF/office/office_exp1_UA2f/ckpts/avbp0yj4/QXxTC'
        ckpts = [os.path.join(trail_dir, f) for f in sorted(os.listdir(trail_dir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.nerf2_network.load_state_dict(ckpt['nerf2_network_state_dict'])


            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,T_0=20,eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']
                                                                                                                                                                                                            


    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        trail_dir = os.path.join(ckptsdir, self.experiment_id, self.id)
        if not os.path.exists(trail_dir):
            os.makedirs(trail_dir)
        model_lst = [x for x in sorted(os.listdir(trail_dir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(trail_dir + '/%s' % model_lst[0])

        ckptname = os.path.join(trail_dir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'nerf2_network_state_dict': self.nerf2_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname



    def train(self):
        """train the model
        """
        
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
        while self.current_iteration <= self.total_iterations:
            total_loss = 0.0
            error = 1e6
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_input, train_label in self.train_iter:
                    if self.current_iteration > self.total_iterations:
                        break

                    train_input, train_label = train_input.to(self.devices), train_label.to(self.devices)
                    if self.dataset_type == "rfid":
                        rays_o, rays_d, tx_o = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:9]
                        predict_spectrum, loss_vae = self.renderer.render_ss(tx_o, rays_o, rays_d)
                        loss = sig2mse(predict_spectrum, train_label.view(-1)) + 0.05 * loss_vae
                    elif self.dataset_type == 'ble':
                        if self.m == 'vae':
                            predict_downlink, loss_vae = self.VAE(train_input)
                            loss = sig2mse(predict_downlink, train_label) + loss_vae * 0.00005
                        elif self.m == 'mlp':
                            predict_downlink = self.MLP(train_input)
                            loss = sig2mse(predict_downlink, train_label)
                        elif self.m == 'unet':
                            train_input = train_input.t()
                            predict_downlink = self.Unet(train_input)
                            loss = sig2mse(predict_downlink, train_label)
                        else:
                            tx_o, rays_o, rays_d = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:]
                            predict_rssi, loss_vae = self.renderer.render_rssi(tx_o, rays_o, rays_d)
                            loss = sig2mse(predict_rssi, train_label.view(-1))                       

                    elif self.dataset_type == 'mimo':
                        if self.m == 'vae':
                            predict_downlink, loss_vae = self.VAE(train_input)
                            loss = sig2mse(predict_downlink, train_label) + loss_vae * 0.00005
                        elif self.m == 'mlp':
                            predict_downlink = self.MLP(train_input)
                            loss = sig2mse(predict_downlink, train_label)
                        elif self.m == 'unet':
                            train_input = train_input.t()
                            predict_downlink = self.Unet(train_input)
                            loss = sig2mse(predict_downlink, train_label)
                        else:
                            uplink, rays_o, rays_d = train_input[:, :52], train_input[:, 52:55], train_input[:, 55:] # each frame captures 26 subcarriers, 26 real + 26 imaginary 
                            predict_downlink, loss_vae = self.renderer.render_csi(uplink, rays_o, rays_d)
                            predict_downlink = torch.concat((predict_downlink.real, predict_downlink.imag), dim=-1)
                            loss = sig2mse(predict_downlink, train_label) + loss_vae * 0.005
                        
                    

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1

                    self.writer.add_scalar('Loss/loss', loss, self.current_iteration)
                    pbar.update(1)
                    pbar.set_description(f"Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str('loss = {:.6f}, lr = {:.6f}'.format(loss.item(), self.optimizer.param_groups[0]['lr']))

                    if self.current_iteration % self.save_freq == 0:
                        ckptname = self.save_checkpoint()
                        pbar.write('Saved checkpoints at {}'.format(ckptname))

                    total_loss += loss.item() / len(self.train_iter)

            print(f"Iteration {self.current_iteration} END")
            print(f"Loss {total_loss}")
            nni.report_intermediate_result(total_loss)
        nni.report_final_result(total_loss)



    def eval_network_spectrum(self):
        """test the model
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        os.makedirs(os.path.join(self.logdir, self.expname, 'pred_spectrum'), exist_ok=True)
        pred2next, gt2next = torch.zeros((0)), torch.zeros((0))
        save_img_idx = 0
        all_ssim = []
        with torch.no_grad():
            for test_input, test_label in self.test_iter:
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                rays_o, rays_d, tx_o = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:9]
                pred_spectrum, _ = self.renderer.render_ss(tx_o, rays_o, rays_d)


                ## save predicted spectrum
                pred_spectrum = pred_spectrum.detach().cpu()
                gt_spectrum = test_label.detach().cpu()
                pred_spectrum = torch.concatenate((pred2next, pred_spectrum), dim=0)
                gt_spectrum = torch.concatenate((gt2next, gt_spectrum), dim=0)
                num_spectrum = len(pred_spectrum) // (360 * 90)
                pred2next = pred_spectrum[num_spectrum*360*90:]
                gt2next = gt_spectrum[num_spectrum*360*90:]

                for i in range(num_spectrum):
                    pred_sepctrum_i = pred_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    gt_spectrum_i = gt_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    pixel_error = np.mean(abs(pred_sepctrum_i - gt_spectrum_i))
                    ssim_i = ssim(pred_sepctrum_i, gt_spectrum_i, data_range=1, multichannel=False)
                    self.logger.info("Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error, ssim_i))
                    paint_spectrum_compare(pred_sepctrum_i, gt_spectrum_i,save_path=os.path.join(self.logdir, self.expname,'pred_spectrum', f'{save_img_idx}.png'))
                    all_ssim.append(ssim_i)
                    self.logger.info("Median SSIM is {:.6f}".format(np.median(all_ssim)))
                    save_img_idx += 1
                    np.savetxt(os.path.join(self.logdir, self.expname, 'all_ssim.txt'), all_ssim, fmt='%.4f')




    def eval_network_rssi(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()
        mse_e = []

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    tx_o, rays_o, rays_d = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:]
                    predict_rssi, _ = self.renderer.render_rssi(tx_o, rays_o, rays_d)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())

                    error = abs(predict_rssi - gt_rssi.reshape(-1))
                    mse = nn.MSELoss()
                    mse_error = np.sqrt(mse(predict_rssi, gt_rssi.reshape(-1)))
                    mse_e.append(mse_error)
                    self.logger.info("Median error:%.2f", torch.median(error))

                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        median = np.median(np.array(mse_e))
        self.logger.info("Total Median MSE:%.2f", median)
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))

    def eval_network_rssi_MLP(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()
        mse_e = []

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    tx_o, rays_o, rays_d = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:]
                    predict_rssi, _ = self.renderer.render_rssi(tx_o, rays_o, rays_d)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())

                    error = abs(predict_rssi - gt_rssi.reshape(-1))
                    mse = nn.MSELoss()
                    mse_error = mse(predict_rssi, gt_rssi.reshape(-1))
                    mse_e.append(mse_error)
                    self.logger.info("Median error:%.2f", torch.median(error))

                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        median = np.median(np.array(mse_e))
        self.logger.info("Total Median MSE:%.2f", median)
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))



    def eval_network_csi(self):
        """test the model and save predicted csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_bs = self.test_set.n_bs    # number of base station antennas
        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        with torch.no_grad():
            for idx, (test_input, test_label) in enumerate(self.test_iter):
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                uplink, rays_o, rays_d = test_input[:, :52], test_input[:, 52:55], test_input[:, 55:]
                predict_downlink, _ = self.renderer.render_csi(uplink, rays_o, rays_d)  # [B, 26]
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                predict_downlink = self.test_set.denormalize_csi(predict_downlink)
                gt_downlink = self.test_set.denormalize_csi(gt_downlink)

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_downlink
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_downlink

                


        all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        snr = csi2snr(all_pred_csi, all_gt_csi)
        self.logger.info("Median SNR:%.2f", torch.median(snr))

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                              'gt_csi': all_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy()})
        
    def eval_network_U(self):
        """test the model and save predicted csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_bs = self.test_set.n_bs    # number of base station antennas
        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        with torch.no_grad():
            for idx, (test_input, test_label) in enumerate(self.test_iter):
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                predict_downlink = self.MLP(test_input)
                predict_downlink = predict_downlink[:, :26] + 1j * predict_downlink[:, 26:]
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                predict_downlink, _ = self.test_set.denormalize_csi(predict_downlink)
                gt_downlink = self.test_set.denormalize_csi(gt_downlink)

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_downlink
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_downlink

                


        all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        snr = csi2snr(all_pred_csi, all_gt_csi)
        self.logger.info("Median SNR:%.2f", torch.median(snr))

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                              'gt_csi': all_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy()})
        
    def eval_network_rssi_VAE(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()
        mse_e = []

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    predict_rssi, _ = self.VAE(test_input)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())

                    error = abs(predict_rssi - gt_rssi.reshape(-1))
                    mse = nn.MSELoss()
                    mse_error = mse(predict_rssi, gt_rssi.reshape(-1))
                    mse_e.append(mse_error)
                    self.logger.info("Median error:%.2f", torch.median(error))

                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        median = np.median(np.array(mse_e))
        self.logger.info("Total Median MSE:%.2f", median)
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))

    def eval_network_rssi_MLP(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.MLP.eval()
        mse_e = []

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    predict_rssi = self.MLP(test_input)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())

                    error = abs(predict_rssi - gt_rssi.reshape(-1))
                    mse = nn.MSELoss()
                    mse_error = mse(predict_rssi, gt_rssi.reshape(-1))
                    mse_e.append(mse_error)
                    self.logger.info("Median error:%.2f", torch.median(error))

                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        median = np.median(np.array(mse_e))
        self.logger.info("Total Median MSE:%.2f", median)
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))

    def eval_network_csi_MLP(self):
        """test the model and save predicted csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_bs = self.test_set.n_bs    # number of base station antennas
        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        with torch.no_grad():
            for idx, (test_input, test_label) in enumerate(self.test_iter):
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                predict_downlink = self.MLP(test_input)  # [B, 26]
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                predict_downlink = self.test_set.denormalize_csi(predict_downlink)
                gt_downlink = self.test_set.denormalize_csi(gt_downlink)

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_downlink
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_downlink

                


        all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        snr = csi2snr(all_pred_csi, all_gt_csi)
        self.logger.info("Median SNR:%.2f", torch.median(snr))

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                              'gt_csi': all_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy()})
    
    def eval_network_csi_VAE(self):
        """test the model and save predicted csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_bs = self.test_set.n_bs    # number of base station antennas
        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        with torch.no_grad():
            for idx, (test_input, test_label) in enumerate(self.test_iter):
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                predict_downlink, _  = self.VAE(test_input)  # [B, 26]
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                predict_downlink = self.test_set.denormalize_csi(predict_downlink)
                gt_downlink = self.test_set.denormalize_csi(gt_downlink)

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_downlink
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_downlink

                


        all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        snr = csi2snr(all_pred_csi, all_gt_csi)
        self.logger.info("Median SNR:%.2f", torch.median(snr))

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                              'gt_csi': all_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy()})
        




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mimo-csi-onlyU.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_type', type=str, default='mimo')
    parser.add_argument('--m', type=str, default='unet')
    parser.add_argument('--n', type=str, default='nni')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    params = nni.get_next_parameter()
    ## backup config file
    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir,'config.yml'))


    worker = NeRF2_Runner(mode=args.mode, dataset_type=args.dataset_type, params=params, m=args.m, n=args.n, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        if args.dataset_type == 'rfid':
            worker.eval_network_spectrum()
        elif args.dataset_type == 'ble':
            if args.m == 'mlp':
                worker.eval_network_rssi_MLP()
            elif args.m == 'vae':
                worker.eval_network_rssi_VAE()
            else:
                worker.eval_network_rssi()
        elif args.dataset_type == 'mimo':
            worker.eval_network_csi()
        elif args.dataset_type == 'newrf':
            worker.eval_network_spectrum_MLP()
        elif args.dataset_type == 'VAE':
            worker.eval_network_csi_VAE()
        elif args.dataset_type == 'MLP':
            worker.eval_network_csi_MLP()
