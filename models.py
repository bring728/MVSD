import torch
import numpy as np
import os
from feature_net import *
from mlp_network import BRDFNet
import glob


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class MVSDModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN'):
        self.cfg = cfg
        self.gpu = gpu
        self.phase = phase
        device = torch.device('cuda:{}'.format(gpu))

        # create feature extraction network
        self.feature_net = ResUNet(out_ch=cfg.feature_dims).to(device)
        self.brdf_net = BRDFNet(cfg).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.feature_net.parameters())
        learnable_params += list(self.brdf_net.parameters())

        self.optimizer = torch.optim.Adam([
            {'params': self.brdf_net.parameters()},
            {'params': self.feature_net.parameters(), 'lr': float(cfg.lr_feature)}],
            lr=float(cfg.lr_mlp))

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if cfg.distributed:
            if phase == 'TRAIN':
                self.brdf_net = torch.nn.parallel.DistributedDataParallel(self.brdf_net, device_ids=[gpu], )
                self.feature_net = torch.nn.parallel.DistributedDataParallel(self.feature_net, device_ids=[gpu], )
                self.switch_to_train()
            elif phase == 'TEST':
                self.switch_to_eval()
            else:
                raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        self.brdf_net.eval()
        self.feature_net.eval()

    def switch_to_train(self):
        self.brdf_net.train()
        self.feature_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'brdf_net': de_parallel(self.brdf_net).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.cfg.distributed:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.brdf_net.load_state_dict(to_load['brdf_net'])
        self.feature_net.load_state_dict(to_load['feature_net'])

    def load_from_ckpt(self, out_folder, load_opt=True, load_scheduler=True):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if len(ckpts) > 0 and (self.cfg.load_ck or self.phase == 'TEST'):
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                step = 999999
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step


class SVNModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN'):
        self.cfg = cfg
        self.gpu = gpu
        self.phase = phase
        device = torch.device('cuda:{}'.format(gpu))

        self.normal_net = NormalNet(cfg).to(device)

        self.optimizer = torch.optim.Adam([{'params': self.normal_net.parameters(), 'lr': float(cfg.lr_normal)}], )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        ckpts = []
        if os.path.exists(experiment):
            ckpts = [os.path.join(experiment, f) for f in sorted(os.listdir(experiment)) if f.endswith('.pth')]
        self.ckpts = ckpts

        if cfg.distributed:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )

        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST':
            self.switch_to_eval()
        else:
            raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        self.normal_net.eval()

    def switch_to_train(self):
        self.normal_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'normal_net': de_parallel(self.normal_net).state_dict()
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.cfg.distributed:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.normal_net.load_state_dict(to_load['normal_net'])

    def load_from_ckpt(self, out_folder, load_opt=True, load_scheduler=True):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if len(ckpts) > 0 and (self.cfg.load_ck or self.phase == 'TEST'):
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                step = 999999
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step


class SVNDModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=False, load_scheduler=True, phase='TRAIN'):
        self.cfg = cfg
        self.gpu = gpu
        self.phase = phase
        self.model_type = cfg.model_type

        device = torch.device('cuda:{}'.format(gpu))

        self.normal_net = NormalNet(cfg).to(device)
        normal_optim_param = {'params': self.normal_net.parameters(), 'lr': float(cfg.lr_normal)}
        self.DL_net = DirectLightingNet(cfg).to(device)
        light_optim_param = {'params': self.DL_net.parameters(), 'lr': float(cfg.lr_light)}

        optim_param = []
        if self.model_type == 'normal' or self.model_type == 'jointly':
            optim_param.append(normal_optim_param)
        if self.model_type == 'light' or self.model_type == 'jointly':
            optim_param.append(light_optim_param)

        self.optimizer = torch.optim.Adam(optim_param, )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_epoch = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if cfg.distributed:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )
            self.DL_net = torch.nn.parallel.DistributedDataParallel(self.DL_net, device_ids=[gpu], )

        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST':
            self.switch_to_eval()
        else:
            raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        self.normal_net.eval()
        self.DL_net.eval()

    def switch_to_train(self):
        if self.model_type == 'normal' or self.model_type == 'jointly':
            self.normal_net.train()
        if self.model_type == 'light' or self.model_type == 'jointly':
            self.DL_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'normal_net': de_parallel(self.normal_net).state_dict(),
                   'DL_net': de_parallel(self.DL_net).state_dict(),
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.cfg.distributed:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.normal_net.load_state_dict(to_load['normal_net'])
        self.DL_net.load_state_dict(to_load['DL_net'])

    def load_from_ckpt(self, out_folder, load_opt=False, load_scheduler=True):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        ckpts_tmp = [x for x in ckpts if 'model_jointly' in x]
        if len(ckpts_tmp) > 0:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                if self.model_type == 'jointly':
                    step = 9999999
                else:
                    step = 0
        else:
            ckpts_tmp = [x for x in ckpts if 'model_light' in x]
            if len(ckpts_tmp) > 0:
                fpath = ckpts[-1]
                self.load_model(fpath, load_opt, load_scheduler)
                step = fpath[-10:-4]
                print('Reloading from {}, starting at step={}'.format(fpath, step))
                if step == 'latest':
                    if self.model_type == 'light':
                        step = 9999999
                    else:
                        step = 0
            else:
                ckpts_tmp = [x for x in ckpts if 'model_normal' in x]
                if len(ckpts_tmp) > 0:
                    fpath = ckpts[-1]
                    self.load_model(fpath, load_opt, load_scheduler)
                    step = fpath[-10:-4]
                    print('Reloading from {}, starting at step={}'.format(fpath, step))
                    if step == 'latest':
                        if self.model_type=='normal':
                            step = 9999999
                        else:
                            step = 0
                else:
                    print('No ckpts found, training from scratch...')
                    step = 0
        return int(step)


class output2env():
    def __init__(self, SGNum, envWidth=16, envHeight=8, isCuda=True):
        self.envWidth = envWidth
        self.envHeight = envHeight

        Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
        El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis=0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32)))

        self.SGNum = SGNum
        if isCuda:
            self.ls = self.ls.cuda()

        self.ls.requires_grad = False

    def fromSGtoIm(self, axis, lamb, weight):
        bn = axis.size(0)
        envRow, envCol = weight.size(2), weight.size(3)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.reshape(bn, self.SGNum, 3, envRow, envCol, 1, 1)
        lamb = lamb.reshape(bn, self.SGNum, 1, envRow, envCol, 1, 1)

        mi = lamb.expand([bn, self.SGNum, 1, envRow, envCol, self.envHeight, self.envWidth]) * \
             (torch.sum(axis.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]) * \
                        self.ls.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]), dim=2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]) * \
                  torch.exp(mi).expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth])

        envmaps = torch.sum(envmaps, dim=1)
        return envmaps

    def output2env(self, axisOrig, lambOrig, weightOrig):
        bn, _, envRow, envCol = weightOrig.size()

        axis = axisOrig

        weight = 0.999 * weightOrig
        weight = torch.tan(np.pi / 2 * weight)

        lambOrig = 0.999 * lambOrig
        lamb = torch.tan(np.pi / 2 * lambOrig)

        envmaps = self.fromSGtoIm(axis, lamb, weight)

        return envmaps, axis, lamb, weight
