import torch
import os
from feature_network import ResUNet
from mlp_network import BRDFNet
import glob


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class MVSDModel(object):
    def __init__(self, cfg, gpu, experiment, load_opt=True, load_scheduler=True):
        self.cfg = cfg
        self.gpu = gpu
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
            self.brdf_net = torch.nn.parallel.DistributedDataParallel(self.brdf_net, device_ids=[gpu], )
            self.feature_net = torch.nn.parallel.DistributedDataParallel(self.feature_net, device_ids=[gpu], )

    def switch_to_eval(self):
        self.brdf_net.eval()
        self.feature_net.eval()

    def switch_to_train(self):
        self.brdf_net.train()
        self.feature_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.brdf_net).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.cfg.distributed:
            to_load = torch.load(filename, map_location={'cuda:0': 'cuda:%d' % self.gpu})
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.brdf_net.load_state_dict(to_load['brdf_net'])
        self.feature_net.load_state_dict(to_load['feature_net'])

    def load_from_ckpt(self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False):
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

        if self.cfg.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.cfg.ckpt_path):  # load the specified ckpt
                ckpts = [self.cfg.ckpt_path]

        if len(ckpts) > 0 and not self.cfg.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step
