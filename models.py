import os
import cv2

import numpy as np

from mlp import *
from cnn import *
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


class NDLModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN', is_DDP=True):
        self.mode = cfg.mode
        self.gpu = gpu
        self.phase = phase
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))

        all_params = []
        self.normal_net = NormalNet(cfg.normal).to(device)
        if self.mode == 'normal' or self.mode == 'finetune':
            all_params.append({'params': self.normal_net.parameters(), 'lr': float(cfg.normal.lr)})

        if self.mode == 'DL' or self.mode == 'finetune':
            self.DL_net = DirectLightingNet(cfg.DL).to(device)
            all_params.append({'params': self.DL_net.parameters(), 'lr': float(cfg.DL.lr)})

        self.optimizer = torch.optim.Adam(all_params)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if self.is_DDP:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )
            if self.mode == 'DL' or self.mode == 'finetune':
                self.DL_net = torch.nn.parallel.DistributedDataParallel(self.DL_net, device_ids=[gpu], )

        if phase == 'TRAIN':
            self.switch_to_train()
        elif phase == 'TEST' or phase == 'ALL':
            self.switch_to_eval()
        else:
            raise Exception('Unrecognized phase for data loader')

    def switch_to_eval(self):
        if self.mode == 'normal' or self.mode == 'finetune':
            self.normal_net.eval()
        if self.mode == 'DL' or self.mode == 'finetune':
            self.DL_net.eval()

    def switch_to_train(self):
        if self.mode == 'normal' or self.mode == 'finetune':
            self.normal_net.train()
        if self.mode == 'DL' or self.mode == 'finetune':
            self.DL_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   }
        if self.mode == 'normal' or self.mode == 'finetune':
            to_save['normal_net'] = de_parallel(self.normal_net).state_dict()
        if self.mode == 'DL' or self.mode == 'finetune':
            to_save['DL_net'] = de_parallel(self.DL_net).state_dict()
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
        print('normal loaded from ', filename)
        if 'DL_net' in to_load:
            self.DL_net.load_state_dict(to_load['DL_net'])
            print('DL loaded from ', filename)

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

        if gpu >= 0:
            device = torch.device('cuda:{}'.format(gpu))
            self.ls = self.ls.to(device)
        self.ls.requires_grad = False

    # Turn SG parameters to environmental maps
    def fromSGtoIm(self, axis, sharp, intensity):
        axis = axis[..., None, None]
        intensity = intensity[..., None, None]
        sharp = sharp[..., None, None]

        mi = sharp * (torch.sum(axis * self.ls, dim=2, keepdim=True) - 1)
        envmaps = intensity * torch.exp(mi)
        envmaps = torch.sum(envmaps, dim=1)
        return envmaps

    def SG_ldr2hdr(self, sharpOrig, intensityOrig):
        intensity = 0.999 * intensityOrig
        intensity = torch.tan(np.pi / 2 * intensity)
        sharp = 0.999 * sharpOrig
        sharp = torch.tan(np.pi / 2 * sharp)
        return sharp, intensity

    def forward_test(self, axis, sharpOrig, intensityOrig, vis):
        intensity = 0.999 * intensityOrig
        intensity = torch.tan(np.pi / 2 * intensity)

        sharp = 0.999 * sharpOrig
        sharp = torch.tan(np.pi / 2 * sharp)

        mi = sharp[..., None, None, None] * (torch.sum(axis[..., None, None] * self.ls[0, :, :, 0, 0], dim=1, keepdim=True) - 1)

        envmaps = intensity[..., None, None, None] * torch.exp(mi)
        envmaps = torch.sum(envmaps, dim=0)
        a = cv2fromtorch(envmaps)
        cv2.imshow('asdf', a)
        cv2.waitKey(0)
        return envmaps


class BRDFModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True, phase='TRAIN', is_DDP=True):
        self.cfg = cfg
        self.gpu = gpu
        self.phase = phase
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))

        root = osp.dirname(osp.dirname(experiment))
        self.normal_net = NormalNet(cfg.normal).to(device)
        self.DL_net = DirectLightingNet(cfg.DL).to(device)
        # NDL_path = osp.join(root, 'stage1', cfg.DL.path, 'model_NDL_latest.pth')
        NDL_path = osp.join(root, 'stage1', cfg.DL.path, 'model_NDL_036000.pth')
        print('read Normal and DL from ', NDL_path)
        if self.is_DDP:
            NDL_ckpt = torch.load(NDL_path, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            NDL_ckpt = torch.load(NDL_path)
        self.normal_net.load_state_dict(NDL_ckpt['normal_net'])
        self.DL_net.load_state_dict(NDL_ckpt['DL_net'])
        if self.is_DDP:
            self.normal_net = torch.nn.parallel.DistributedDataParallel(self.normal_net, device_ids=[gpu], )
            self.DL_net = torch.nn.parallel.DistributedDataParallel(self.DL_net, device_ids=[gpu], )
        self.normal_net.eval()
        self.DL_net.eval()

        # create feature extraction network
        self.feature_net = Context_ResUNet(cfg.BRDF.context_feature).to(device)
        self.brdf_net = MultiViewAggregation(cfg).to(device)
        self.brdf_refine_net = BRDFRefineNet(cfg).to(device)

        all_params = [
            {'params': self.feature_net.parameters(), 'lr': float(cfg.BRDF.context_feature.lr)},
            {'params': self.brdf_net.parameters(), 'lr': float(cfg.BRDF.aggregation.lr)},
            {'params': self.brdf_refine_net.parameters(), 'lr': float(cfg.BRDF.refine.lr)},
        ]
        # stage = osp.basename(osp.dirname(experiment))
        # if stage == 'stage3':
        #     self.feature_GL_net =
        # count_parameters(self.feature_net)
        # optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(all_params)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=int(cfg.lrate_decay_steps),
                                                         gamma=float(cfg.lrate_decay_factor))

        self.start_step = self.load_from_ckpt(experiment, load_opt=load_opt, load_scheduler=load_scheduler)

        if self.is_DDP:
            self.feature_net = torch.nn.parallel.DistributedDataParallel(self.feature_net, device_ids=[gpu], )
            self.brdf_net = torch.nn.parallel.DistributedDataParallel(self.brdf_net, device_ids=[gpu], )
            self.brdf_refine_net = torch.nn.parallel.DistributedDataParallel(self.brdf_refine_net, device_ids=[gpu], )

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

    def load_from_ckpt(self, out_folder, load_opt=True, load_scheduler=True, ):
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
        return int(step)


if __name__ == '__main__':
    sg2env = SG2env(4, envWidth=16, envHeight=8, gpu=-1)
    Az = ((np.arange(16) + 0.5) / 16 - 0.5) * 2 * np.pi
    El = ((np.arange(8) + 0.5) / 8) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis=0)

    axis = torch.from_numpy(np.stack([ls[:, 2, 4], ls[:, 4, 4], ls[:, 2, 8], ls[:, 4, 8]], axis=0))
    sharp = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.9]))
    intensity = torch.from_numpy(np.array([0.0, 0.0, 0.0, 0.2]))
    sg2env.forward_test(axis, sharp, intensity, None)
