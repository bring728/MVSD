import torch
import cv2
from utils import *
import torch.nn.functional as F
from collections import OrderedDict
from torch.cuda.amp.autocast_mode import autocast


def sample_view(data, num_view):
    data['rgb'] = data['rgb'][:, :num_view]
    data['mask'] = data['mask'][:, :num_view]
    data['depth_est'] = data['depth_est'][:, :num_view]
    data['depth_gt'] = data['depth_gt'][:, :num_view]
    data['depth_norm'] = data['depth_norm'][:, :num_view]
    data['conf'] = data['conf'][:, :num_view]
    data['cam'] = data['cam'][:, :num_view]
    data['c2w'] = data['c2w'][:, :num_view]


def model_forward(stage, phase, curr_model, helper_dict, data, cfg, scalars_to_log, save_image_flag):
    with autocast(enabled=cfg.autocast):
        pred = {}
        total_loss = None
        if stage == '1-1':
            normal_pred = curr_model.normal_net(data['input'])
            normal_mse_err = img2mse(normal_pred, data['normal_gt'], data['mask'][:, 1:, ...])
            normal_ang_err = img2angerr(normal_pred, data['normal_gt'], data['mask'][:, 1:, ...])
            scalars_to_log['train/normal_mse_err'] = normal_mse_err.item()
            scalars_to_log['train/normal_ang_err'] = normal_ang_err.item()
            total_loss = cfg.lambda_mse * normal_mse_err + cfg.lambda_ang * normal_ang_err
            scalars_to_log['train/total_loss'] = total_loss.item()
            pred['normal'] = normal_pred

        elif stage == '1-2':
            with torch.no_grad():
                pred['normal'] = curr_model.normal_net(data['input'])
            rgbdcn = torch.cat([data['input'], 0.5 * (pred['normal'] + 1)], dim=1)
            axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
            if phase == 'output':
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
                # vis_beta_loss = torch.mean(torch.log10(0.1 + vis) + torch.log10(0.1 + 1. - vis) + 1.)
                scalars_to_log['train/env_scaled_loss'] = env_scaled_loss.item()
                scalars_to_log['train/vis_beta_loss'] = vis_beta_loss.item()
                total_loss = cfg.lambda_vis_prior * vis_beta_loss + env_scaled_loss
                scalars_to_log['train/total_loss'] = total_loss.item()

        elif stage == '2':
            sample_view(data, 7 + np.random.choice(3, 1)[0])
            target_rgbdc = torch.cat([data['rgb'][:, 0], data['depth_norm'][:, 0], data['conf'][:, 0]], dim=1)

            with torch.no_grad():
                pred['normal'] = curr_model.normal_net(target_rgbdc)
                rgbdcn = torch.cat([target_rgbdc, 0.5 * (pred['normal'] + 1)], dim=1)
                axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
                bn, _, _, rows, cols = axis.shape
                DL_pred = torch.cat([axis, sharpness, intensity * vis], dim=2).reshape((bn, -1, rows, cols))
                pred['DL'] = DL_pred
            pixel_batch = helper_dict['pixels']
            if save_image_flag:
                DL = pred['DL'].reshape(bn, cfg.DL.SGNum, 7, cfg.DL.env_rows, cfg.DL.env_cols)
                envmaps_pred = helper_dict['sg2env'].forward(DL[:, :, :3], DL[:, :, 3:4], DL[:, :, 4:], None)
                pred['envmaps'] = envmaps_pred

            if cfg.BRDF.except_feature:
                if cfg.BRDF.gt:
                    rgb_feat, viewdir, proj_err = compute_projection(pixel_batch, data['cam'], data['c2w'], data['depth_gt'], data['rgb'],
                                                                     None)
                else:
                    rgb_feat, viewdir, proj_err = compute_projection(pixel_batch, data['cam'], data['c2w'], data['depth_est'], data['rgb'],
                                                                     None)
                normal_pred = pred['normal'].permute(0, 2, 3, 1)[:, :, :, None]
                DL_target = F.grid_sample(pred['DL'], pixel_batch[..., 3:][None].expand([bn, -1, -1, -1]), align_corners=False, mode='nearest')

                brdf = curr_model.brdf_net(rgb_feat, viewdir, proj_err, normal_pred, DL_target.permute(0, 2, 3, 1)[:, :, :, None]).permute(0, 3, 1, 2)
                refine_input = torch.cat([target_rgbdc, brdf], dim=1)
                brdf = curr_model.brdf_refine_net(refine_input)

            else:
                bn, vn, _, h, w = data['rgb'].shape
                if cfg.BRDF.context_feature.input == 'rgbdc':
                    rgbdc = torch.cat([data['rgb'], data['depth_norm'], data['conf']], dim=2).reshape([bn * vn, 5, h, w])
                    featmaps = curr_model.feature_net(rgbdc)
                else:
                    rgb = data['rgb'].reshape([bn * vn, 3, h, w])
                    featmaps = curr_model.feature_net(rgb)

                if cfg.BRDF.gt:
                    rgb_feat, viewdir, proj_err = compute_projection(pixel_batch, data['cam'], data['c2w'], data['depth_gt'], data['rgb'],
                                                                     featmaps)
                else:
                    rgb_feat, viewdir, proj_err = compute_projection(pixel_batch, data['cam'], data['c2w'], data['depth_est'], data['rgb'],
                                                                     featmaps)
                normal_pred = pred['normal'].permute(0, 2, 3, 1)[:, :, :, None]
                DL_target = F.grid_sample(pred['DL'], pixel_batch[..., 3:][None].expand([bn, -1, -1, -1]), align_corners=False, mode='nearest')

                brdf = curr_model.brdf_net(rgb_feat, viewdir, proj_err, normal_pred, DL_target.permute(0, 2, 3, 1)[:, :, :, None]).permute(0, 3, 1, 2)
                if cfg.BRDF.refine.use:
                    refine_input = torch.cat([target_rgbdc, brdf], dim=1)
                    if cfg.BRDF.refine.context_residual:
                        _, c, h, w = featmaps.shape
                        target_feat = featmaps.reshape([bn, vn, c, h, w])[:, 0]
                        brdf = curr_model.brdf_refine_net(refine_input, target_feat)
                    else:
                        brdf = curr_model.brdf_refine_net(refine_input)

            segBRDF = data['mask'][:, :1]
            albedo_pred = brdf[0]
            rough_pred = brdf[1]
            pred['rough'] = rough_pred
            if cfg.BRDF.conf.use:
                conf_pred = brdf[2]
                pred['conf'] = conf_pred
                if cfg.confloss == 'linear':
                    conf_loss = img2L1Loss(conf_pred, 1.0, segBRDF)
                else:
                    conf_loss = torch.clamp(-torch.log(conf_pred + 0.1), min=0.0)
                scalars_to_log['train/conf_loss'] = conf_loss.item()
            else:
                conf_pred = None
                conf_loss = 0.0
                pred['conf'] = torch.ones_like(rough_pred)

            albedo_pred_scaled = LSregress(albedo_pred.detach() * segBRDF, data['albedo_gt'] * segBRDF, albedo_pred)
            albedo_pred_scaled = torch.clamp(albedo_pred_scaled, 0, 1)
            pred['albedo'] = albedo_pred_scaled

            albedo_mse_err = img2mse(albedo_pred_scaled, data['albedo_gt'], segBRDF, conf_pred)
            rough_mse_err = img2mse(rough_pred, data['rough_gt'], segBRDF, conf_pred)

            scalars_to_log['train/albedo_mse_err'] = albedo_mse_err.item()
            scalars_to_log['train/rough_mse_err'] = rough_mse_err.item()
            total_loss = cfg.BRDF.lambda_albedo * albedo_mse_err + cfg.BRDF.lambda_rough * rough_mse_err + cfg.BRDF.lambda_conf * conf_loss
            scalars_to_log['train/total_loss'] = total_loss.item()
        else:
            raise Exception('stage error')

    return total_loss, pred


def compute_projection(pixel_batch, int_list, c2w_list, depth_list, im_list, featmaps):
    bn, vn, _, h, w = depth_list.shape
    w2c_list = torch.inverse(c2w_list)
    pixel_depth = depth_list[:, 0, 0, ..., None, None]

    cam_coord = pixel_depth * torch.inverse(int_list[:, None, None, 0]) @ pixel_batch[None, :, :, :3, None]
    # because cam_0 is world
    world_coord = torch.cat([cam_coord, torch.ones_like(cam_coord[:, :, :, :1, :])], dim=-2)

    # get projection coord
    cam_coord_k = (w2c_list[:, :, None, None] @ world_coord[:, None])[..., :3, :]
    pixel_coord_k = (int_list[:, :, None, None] @ cam_coord_k)[..., 0]
    pixel_depth_k_est = torch.clamp(pixel_coord_k[..., 2:3], min=1e-5)
    pixel_coord_k_est = pixel_coord_k[..., :2] / pixel_depth_k_est

    resize_factor = torch.tensor([w, h]).to(pixel_coord_k_est.device)[None, None, None, None, :]
    pixel_coord_k_norm = (2 * pixel_coord_k_est / resize_factor - 1.).reshape([bn * vn, h, w, 2])
    pixel_rgbd_k = F.grid_sample(torch.cat([im_list, depth_list], dim=2).reshape([bn * vn, 4, h, w]), pixel_coord_k_norm,
                                 align_corners=False, mode='nearest').reshape([bn, vn, 4, h, w])
    proj_err = pixel_depth_k_est - pixel_rgbd_k[:, :, -1, ..., None]
    # print(torch.mean(torch.abs(pixel_rgbd_k[0,0, -1] - depth_list[0, 0, 0])))
    # print(torch.mean(torch.abs(pixel_rgbd_k[0, 0, :3] - im_list[0, 0, :])))

    rgb_sampled = pixel_rgbd_k[:, :, :3]
    if featmaps is None:
        rgb_feat_sampled = rgb_sampled.permute(0, 3, 4, 1, 2)
    else:
        feat_sampled = F.grid_sample(featmaps, pixel_coord_k_norm, align_corners=False, mode='nearest').reshape([bn, vn, -1, h, w])
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=2).permute(0, 3, 4, 1, 2)
    viewdir = F.normalize((c2w_list[:, :, None, None, :3, :3] @ cam_coord_k)[..., 0], dim=-1)
    # weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
    # weight = weight / (torch.sum(weight, dim=1, keepdim=True) + TINY_NUMBER)
    # while True:
    #     h = np.random.randint(0, 480)
    #     w = np.random.randint(0, 640)
    #     print(weight[0, :4, h, w, 0])
    #     coord = pixel_coord_k_est[0, :4, h, w]
    #     rgb_list = []
    #     rgb = (im_list[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8).copy()
    #     rgb1 = cv2.circle(rgb[0], (int(coord[0, 0]), int(coord[0, 1])), 5, (255, 255, 0), -1)
    #     rgb2 = cv2.circle(rgb[1], (int(coord[1, 0]), int(coord[1, 1])), 5, (255, 255, 0), -1)
    #     rgb3 = cv2.circle(rgb[2], (int(coord[2, 0]), int(coord[2, 1])), 5, (255, 255, 0), -1)
    #     rgb4 = cv2.circle(rgb[3], (int(coord[3, 0]), int(coord[3, 1])), 5, (255, 255, 0), -1)
    #     rgb_list.append(rgb1)
    #     rgb_list.append(rgb2)
    #     rgb_list.append(rgb3)
    #     rgb_list.append(rgb4)
    #     rgb = cv2.hconcat(rgb_list)
    #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #     cv2.imshow('asd', rgb)
    #     cv2.waitKey(0)
    return rgb_feat_sampled, viewdir.permute(0, 2, 3, 1, 4), proj_err.permute(0, 2, 3, 1, 4)


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
