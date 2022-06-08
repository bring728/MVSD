import torch
import numpy as np
import os
from network import *
import os.path as osp


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MonoNormalModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN', is_DDP=True):
        self.gpu = gpu
        self.phase = phase
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))

        self.normal_net = NormalNet(cfg.normal).to(device)
        normal_optim_param = {'params': self.normal_net.parameters(), 'lr': float(cfg.lr)}

        optim_param = []
        optim_param.append(normal_optim_param)

        self.optimizer = torch.optim.Adam(optim_param, )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if self.is_DDP:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )

        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST' or phase == 'ALL':
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
                   'normal_net': de_parallel(self.normal_net).state_dict(),
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.is_DDP:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])
        self.normal_net.load_state_dict(to_load['normal_net'])

    def load_from_ckpt(self, out_folder, load_opt=False, load_scheduler=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if len(ckpts) > 0:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                step = 9999999
        else:
            print('No ckpts found, training from scratch...')
            step = 0
        return int(step)


class MonoDirectLightModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN', is_DDP=True):
        self.gpu = gpu
        self.phase = phase
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))

        root = osp.dirname(osp.dirname(experiment))

        self.normal_net = NormalNet(cfg.normal).to(device)
        if self.is_DDP:
            normal_ckpt = torch.load(osp.join(osp.join(root, 'stage1-1', cfg.normal.path), 'model_normal_latest.pth'),
                                     map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            normal_ckpt = torch.load(osp.join(osp.join(root, 'stage1-1', cfg.normal.path), 'model_normal_latest.pth'))
        self.normal_net.load_state_dict(normal_ckpt['normal_net'])

        self.DL_net = DirectLightingNet(cfg.DL).to(device)
        light_optim_param = {'params': self.DL_net.parameters(), 'lr': float(cfg.lr)}

        optim_param = []
        optim_param.append(light_optim_param)

        self.optimizer = torch.optim.Adam(optim_param, )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if self.is_DDP:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )
            self.DL_net = torch.nn.parallel.DistributedDataParallel(self.DL_net, device_ids=[gpu], )

        self.normal_net.eval()
        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST' or phase == 'ALL':
            self.switch_to_eval()
        else:
            raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        self.DL_net.eval()

    def switch_to_train(self):
        self.DL_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'DL_net': de_parallel(self.DL_net).state_dict(),
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.is_DDP:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])
        self.DL_net.load_state_dict(to_load['DL_net'])

    def load_from_ckpt(self, out_folder, load_opt=False, load_scheduler=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''
        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if len(ckpts) > 0:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                step = 9999999
        else:
            print('No ckpts found, training from scratch...')
            step = 0
        return int(step)


class SG2env():
    def __init__(self, SGNum, envWidth=16, envHeight=8, gpu=0):
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
        self.ls = torch.from_numpy(ls.astype(np.float32))

        self.SGNum = SGNum

        device = torch.device('cuda:{}'.format(gpu))
        self.ls = self.ls.to(device)
        self.ls.requires_grad = False

    # Turn SG parameters to environmental maps
    def fromSGtoIm(self, axis, sharp, intensity, vis):
        bn, _1, _2, envRow, envCol = axis.size()

        axis = axis[..., None, None]
        intensity = intensity[..., None, None]
        sharp = sharp.reshape(bn, self.SGNum, 1, envRow, envCol, 1, 1)

        mi = sharp * (torch.sum(axis * self.ls, dim=2, keepdim=True) - 1)

        if vis is None:
            envmaps = intensity * torch.exp(mi)
        else:
            vis = vis.reshape(bn, self.SGNum, 1, envRow, envCol, 1, 1)
            envmaps = vis * intensity * torch.exp(mi)

        envmaps = torch.sum(envmaps, dim=1)
        return envmaps

    def forward(self, axis, sharpOrig, intensityOrig, vis):
        intensity = 0.999 * intensityOrig
        intensity = torch.tan(np.pi / 2 * intensity)

        sharp = 0.999 * sharpOrig
        sharp = torch.tan(np.pi / 2 * sharp)

        envmaps = self.fromSGtoIm(axis, sharp, intensity, vis)
        return envmaps


class BRDFModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN', is_DDP=True):
        self.cfg = cfg
        self.gpu = gpu
        self.phase = phase
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))

        # root = osp.dirname(osp.dirname(experiment))
        # self.normal_net = NormalNet(cfg.normal).to(device)
        # self.DL_net = DirectLightingNet(cfg.DL).to(device)
        # if cfg.normal.path == None:
        #     cfg.normal.path = osp.join(osp.join(root, 'stage1-1'), sorted(os.listdir(osp.join(root, 'stage1-1')))[-1])
        # if cfg.DL.path == None:
        #     cfg.DL.path = osp.join(osp.join(root, 'stage1-1'), sorted(os.listdir(osp.join(root, 'stage1-2')))[-1])
        # if self.is_DDP:
        #     normal_ckpt = torch.load(osp.join(root, 'stage1-1', cfg.normal.path, 'model_normal_latest.pth'),
        #                              map_location={'cuda:0': 'cuda:%d' % self.gpu})
        #     DL_ckpt = torch.load(osp.join(root, 'stage1-2', cfg.DL.path, 'model_DL_latest.pth'),
        #                          map_location={'cuda:0': 'cuda:%d' % self.gpu})
        # else:
        #     normal_ckpt = torch.load(osp.join(root, 'stage1-1', cfg.normal.path, 'model_normal_latest.pth'))
        #     DL_ckpt = torch.load(osp.join(root, 'stage1-2', cfg.DL.path, 'model_DL_latest.pth'))
        # self.normal_net.load_state_dict(normal_ckpt['normal_net'])
        # self.DL_net.load_state_dict(DL_ckpt['DL_net'])

        # create feature extraction network
        self.feature_net = ResUNet(cfg).to(device)
        if cfg.BRDF.net_type == 'mlp':
            self.brdf_net = BRDF_mlp(cfg).to(device)
        else:
            self.brdf_net = BRDF_transformer(cfg).to(device)
        self.brdf_refine_net = BRDFRefineNet(cfg.BRDF).to(device)

        # count_parameters(self.feature_net)

        # optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam([
            {'params': self.brdf_net.parameters(), 'lr': float(cfg.lr_mlp)},
            {'params': self.feature_net.parameters(), 'lr': float(cfg.lr_feature)},
            {'params': self.brdf_refine_net.parameters(), 'lr': float(cfg.lr_refine)}])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if self.is_DDP:
            # self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )
            # self.DL_net = torch.nn.parallel.DistributedDataParallel(self.DL_net, device_ids=[gpu], )
            self.feature_net = torch.nn.parallel.DistributedDataParallel(self.feature_net, device_ids=[gpu], )
            self.brdf_net = torch.nn.parallel.DistributedDataParallel(self.brdf_net, device_ids=[gpu], )
            self.brdf_refine_net = torch.nn.parallel.DistributedDataParallel(self.brdf_refine_net, device_ids=[gpu], )

        # self.normal_net.eval()
        # self.DL_net.eval()
        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST' or phase == 'ALL':
            self.switch_to_eval()
        else:
            raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        self.brdf_net.eval()
        self.feature_net.eval()
        self.brdf_refine_net.eval()

    def switch_to_train(self):
        self.brdf_net.train()
        self.feature_net.train()
        self.brdf_refine_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'brdf_net': de_parallel(self.brdf_net).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict(),
                   'brdf_refine_net': de_parallel(self.brdf_refine_net).state_dict()
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.phase == 'TRAIN' and self.is_DDP:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)
        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])
        self.brdf_net.load_state_dict(to_load['brdf_net'])
        self.feature_net.load_state_dict(to_load['feature_net'])
        self.brdf_refine_net.load_state_dict(to_load['brdf_refine_net'])

    # def update_optim(self, global_step):
    #     n = int(global_step / self.cfg.lrate_decay_steps)
    #     self.optimizer.param_groups[0]['lr'] *= self.cfg.lrate_decay_factor ** n

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

        if len(ckpts) > 0:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = fpath[-10:-4]
            print('Reloading from {}, starting at step={}'.format(fpath, step))
            if step == 'latest':
                step = 9999999
            # else:
            #     step = int(step)
            #     self.update_optim(step)
        else:
            print('No ckpts found, training from scratch...')
            step = 0
        return step
