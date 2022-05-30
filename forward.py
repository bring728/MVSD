import torch
from utils import *
import torch.nn.functional as F
from collections import OrderedDict
from torch.cuda.amp.autocast_mode import autocast


def model_forward(stage, phase, curr_model, helper_dict, data, cfg, scalars_to_log, save_image_flag):
    with autocast(enabled=cfg.autocast):
        pred = {}
        total_loss = None
        if stage == '1-1':
            normal_pred = curr_model.normal_net(data['input'])

            normal_mse_err = img2mse(normal_pred, data['normal_gt'], data['mask'][:, 1:, ...])
            normal_ang_err = img2angerr(normal_pred, data['normal_gt'], data['mask'][:, 1:, ...])
            total_loss = cfg.lambda_mse * normal_mse_err + cfg.lambda_ang * normal_ang_err

            scalars_to_log['train/normal_mse_err'] = normal_mse_err.item()
            scalars_to_log['train/normal_ang_err'] = normal_ang_err.item()
            pred['normal'] = normal_pred
        elif stage == '1-2':
            with torch.no_grad():
                normal_pred = curr_model.normal_net(data['input'])
            pred['normal'] = normal_pred
            rgbdcn = torch.cat([data['input'], 0.5 * (normal_pred + 1)], dim=1)
            axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
            if phase == 'ALL':
                bn, _, _, rows, cols = axis.shape
                DL_pred = torch.cat([axis, sharpness, intensity * vis], dim=2).reshape((bn, -1, rows, cols))
                pred['DL'] = DL_pred
            else:
                segBRDF = F.adaptive_avg_pool2d(data['mask'][:, :1, ...], (cfg.DL.env_rows, cfg.DL.env_cols))
                notDarkEnv = (torch.mean(data['envmaps_gt'], dim=(1, 4, 5)) > 0.001).float()[:, None]
                segEnvBatch = (segBRDF * data['envmapsInd'])[..., None, None]
                segEnvBatch = (segEnvBatch * notDarkEnv[..., None, None]).expand_as(data['envmaps_gt'])

                envmaps_pred = helper_dict['sg2env'].forward(axis, sharpness, intensity, vis)
                pred['envmaps'] = envmaps_pred
                envmaps_pred_scaled = LSregress(envmaps_pred.detach() * segEnvBatch, data['envmaps_gt'] * segEnvBatch, envmaps_pred)

                env_scaled_loss = img2log_mse(envmaps_pred_scaled, data['envmaps_gt'], segEnvBatch)
                vis_beta_loss = torch.mean(torch.log(0.1 + vis) + torch.log(0.1 + 1. - vis) + 2.20727)  # from neural volumes
                scalars_to_log['train/env_scaled_loss'] = env_scaled_loss.item()
                scalars_to_log['train/vis_beta_loss'] = vis_beta_loss.item()

                total_loss = cfg.lambda_vis_prior * vis_beta_loss + env_scaled_loss
                scalars_to_log['train/total_loss'] = total_loss.item()
        elif stage == '2':
            # with torch.no_grad():
            #     rgbdc_target = rgbdc[:1]
            #     normal_pred = curr_model.normal_net(rgbdc_target)
            #     pred['normal'] = normal_pred
            #     rgbdcn_target = torch.cat([rgbdc_target, 0.5 * (normal_pred + 1)], dim=1)
            #     axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn_target)
            #     DL_target = torch.cat([axis, sharpness, intensity * vis], dim=2).reshape((1, -1, axis.shape[-2], axis.shape[-1]))
            if cfg.BRDF.input_feature == 'rgbdc':
                rgbdc = torch.cat([data['rgb'], data['depth_norm'], data['conf']], dim=2)[0]
                featmaps = curr_model.feature_net(rgbdc)
            else:
                featmaps = curr_model.feature_net(data['rgb'][0])

            # num_view = rgbdc.shape[0]
            # num_batch = int(1.0 * cfg.ray_batch * cfg.num_view_min / num_view)
            # select_inds = rng.choice(cfg.imWidth * cfg.imHeight, size=(num_batch,), replace=False)
            # pixel_batch = pixels[select_inds]
            if save_image_flag:
                DL = data['DL'].reshape(1, cfg.DL.SGNum, 7, 30, 40)
                envmaps_pred = helper_dict['sg2env'].forward(DL[:, :, :3], DL[:, :, 3:4], DL[:, :, 4:], None)
                pred['envmaps'] = envmaps_pred

            pixel_batch = helper_dict['pixels']
            rgb_feat, viewdir, proj_err = compute_projection(pixel_batch, data['cam'][0], data['c2w'][0], data['depth'][0], data['rgb'][0],
                                                             featmaps)
            normal_pred = data['normal'].permute(2, 3, 0, 1)
            DL_target = F.grid_sample(data['DL'], pixel_batch[..., 3:][None], align_corners=False).permute(2, 3, 0, 1)
            brdf_feature = curr_model.brdf_net(rgb_feat, viewdir, proj_err, normal_pred, DL_target).permute(2, 0, 1)[None]

            if cfg.depth_gt:
                refine_input = torch.cat(
                    [data['target_depth_norm'][:1, :1], torch.ones_like(data['target_depth_norm'][:1, :1]), brdf_feature], dim=1)
            else:
                refine_input = torch.cat([data['target_depth_norm'][:1, :1], data['conf'][0, :1], brdf_feature], dim=1)
            if cfg.BRDF.refine_input != 'dc':
                refine_input = torch.cat([data['rgb'][0, :1], refine_input], dim=1)
            brdf = curr_model.brdf_refine_net(refine_input)

            # brdf = brdf.reshape(4, cfg.imHeight, cfg.imWidth)
            albedo_pred = brdf[:, :3]
            rough_pred = brdf[:, 3:-1]
            conf_pred = brdf[:, -1:]
            pred['rough'] = rough_pred
            pred['conf'] = conf_pred

            segBRDF = data['mask'][:, :1]
            albedo_pred_scaled = LSregress(albedo_pred.detach() * segBRDF, data['albedo_gt'] * segBRDF, albedo_pred)
            albedo_pred_scaled = torch.clamp(albedo_pred_scaled, 0, 1)
            pred['albedo'] = albedo_pred_scaled

            albedo_mse_err = img2mse(albedo_pred_scaled, data['albedo_gt'], segBRDF, conf_pred)
            rough_mse_err = img2mse(rough_pred, data['rough_gt'], segBRDF, conf_pred)
            conf_loss = img2L1Loss(conf_pred, 1.0, segBRDF)
            scalars_to_log['train/albedo_mse_err'] = albedo_mse_err.item()
            scalars_to_log['train/rough_mse_err'] = rough_mse_err.item()
            scalars_to_log['train/conf_loss'] = conf_loss.item()

            total_loss = cfg.lambda_albedo * albedo_mse_err + cfg.lambda_rough * rough_mse_err + cfg.lambda_conf * conf_loss
            scalars_to_log['train/total_loss'] = total_loss.item()
        else:
            raise Exception('stage error')

    return total_loss, pred


def compute_projection(pixel_batch, int_list, c2w_list, depth_list, im_list, featmaps):
    h, w = depth_list.shape[2:]
    w2c_list = torch.inverse(c2w_list)
    pixel_depth = depth_list[0, 0][..., None, None]  # reshape order is x - y! (last index is priority)

    cam_coord = pixel_depth * torch.inverse(int_list[0]) @ pixel_batch[..., :3, None]
    # because cam_0 is world
    world_coord = torch.cat([cam_coord, torch.ones_like(cam_coord[:, :, :1, :])], dim=-2)

    # get projection coord
    cam_coord_k = (w2c_list[:, None, None] @ world_coord[None])[..., :3, :]
    pixel_coord_k = (int_list[:, None, None] @ cam_coord_k)[..., 0]
    pixel_depth_k = torch.clamp(pixel_coord_k[..., 2:3], min=1e-5)
    pixel_coord_k = pixel_coord_k[..., :2] / pixel_depth_k

    resize_factor = torch.tensor([w, h]).to(pixel_coord_k.device)[None, None, None, :]
    pixel_coord_k_norm = (2 * pixel_coord_k / resize_factor - 1.)

    # get depth error
    pixel_rgbd_k = F.grid_sample(torch.cat([im_list, depth_list], dim=1), pixel_coord_k_norm, align_corners=False)
    proj_err = pixel_depth_k - pixel_rgbd_k[:, -1, ..., None]
    # torch.nonzero(torch.where(proj_err > 1, 1, 0))
    # torch.nonzero(torch.where(proj_err < -1, 1, 0))

    rgb_sampled = pixel_rgbd_k[:, :3]
    feat_sampled = F.grid_sample(featmaps, pixel_coord_k_norm, align_corners=False)
    rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=1).permute(2, 3, 0, 1)
    viewdir = F.normalize((c2w_list[:, None, None, :3, :3] @ cam_coord_k)[..., 0], dim=-1)
    return rgb_feat_sampled, viewdir.permute(1, 2, 0, 3), proj_err.permute(1, 2, 0, 3)


def decompose_single_image(model, gpu, chunk_size, pixels, val_data):
    ret = OrderedDict()
    ret['albedo'] = []
    ret['normal'] = []
    ret['roughness'] = []

    depth_list = val_data['depth_list'][0].cuda(gpu)
    int_list = val_data['int_list'][0].cuda(gpu)
    c2w_list = val_data['c2w_list'][0].cuda(gpu)
    im_list = val_data['im_list'][0].cuda(gpu)
    target_gt = val_data['target_gt'][0].cuda(gpu)
    ch, h, w = target_gt.shape
    depth_gt_flat = depth_list[0].reshape(-1)

    featmaps = model.feature_net(im_list)
    N_rays = pixels.shape[0]
    for i in range(0, N_rays, chunk_size):
        pixel_batch = pixels[i:i + chunk_size]
        depth_gt = depth_gt_flat[i:i + chunk_size][:, None]

        rgb_feat_viewdir_err = compute_projection(pixel_batch, depth_gt, int_list, c2w_list, depth_list, im_list, featmaps)
        brdf = model.brdf_net(rgb_feat_viewdir_err)

        albedo = 0.5 * (brdf[..., :3] + 1)

        x_orig = brdf[..., 3:6]
        norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
        normal = x_orig / torch.clamp(norm, min=1e-6)

        roughness = brdf[..., 6:]

        ret['albedo'].append(albedo.cpu())
        ret['normal'].append(normal.cpu())
        ret['roughness'].append(roughness.cpu())

    for k in ret:
        tmp = torch.cat(ret[k], dim=0).reshape((h, w, -1))
        ret[k] = tmp.permute(2, 0, 1)

    albedo = ret['albedo']
    segBRDF = target_gt[8:9].cpu()
    albedo_gt = target_gt[:3].cpu()
    ret['albedo'] = LSregress(albedo * segBRDF, albedo_gt * segBRDF, albedo)

    ret['albedo_gt'] = albedo_gt
    ret['normal_gt'] = target_gt[3:6].cpu()
    ret['roughness_gt'] = target_gt[6:7].cpu()
    ret['segBRDF'] = segBRDF
    ret['segAll'] = target_gt[9:].cpu()
    return ret
