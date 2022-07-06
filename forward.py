import torch
import cv2
from utils import *
import torch.nn.functional as F
from collections import OrderedDict
from torch.cuda.amp.autocast_mode import autocast


def sample_view(data, num_view, gt=False):
    data['rgb'] = data['rgb'][:, :num_view]
    data['mask'] = data['mask'][:, :num_view]
    data['depth_est'] = data['depth_est'][:, :num_view]
    if gt:
        data['depth_gt'] = data['depth_gt'][:, :num_view]
    data['depth_norm'] = data['depth_norm'][:, :num_view]
    # data['conf'] = data['conf'][:, :num_view]
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
            sample_view(data, 7 + np.random.choice(3, 1)[0], gt=cfg.BRDF.gt)
            target_rgbdc = torch.cat([data['rgb'][:, 0], data['depth_norm'][:, 0], data['conf']], dim=1)
            bn, _, rows, cols = target_rgbdc.shape
            with torch.no_grad():
                data['normal'] = curr_model.normal_net(target_rgbdc)
                # rgbdcn = torch.cat([target_rgbdc, 0.5 * (data['normal'] + 1)], dim=1)
                # axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
                # bn, _, _, rows, cols = axis.shape
                # DL_pred = torch.cat([axis, sharpness, intensity * vis], dim=2).reshape((bn, -1, rows, cols))
                # data['DL'] = DL_pred

            pixels = helper_dict['pixels']
            pixels_norm = helper_dict['pixels_norm']
            if save_image_flag:
                DL = data['DL'].reshape(bn, cfg.DL.SGNum, 7, cfg.DL.env_rows, cfg.DL.env_cols)
                envmaps_pred = helper_dict['sg2env'].forward(DL[:, :, :3], DL[:, :, 3:4], DL[:, :, 4:], None)
                data['envmaps'] = envmaps_pred
            bn, vn, _, h, w = data['rgb'].shape
            if cfg.BRDF.context_feature.input == 'rgbdc':
                featmaps = curr_model.feature_net(target_rgbdc)
            else:
                rgb = data['rgb'][:, 0]
                featmaps = curr_model.feature_net(rgb)

            if cfg.BRDF.gt:
                rgb_sampled, viewdir, proj_err = compute_projection(pixels, data['cam'], data['c2w'], data['depth_gt'], data['rgb'])
            else:
                rgb_sampled, viewdir, proj_err = compute_projection(pixels, data['cam'], data['c2w'], data['depth_est'], data['rgb'])
            normal_pred = data['normal'].permute(0, 2, 3, 1)[:, :, :, None]
            DL_target = F.grid_sample(data['DL'], pixels_norm, align_corners=False, mode='nearest').permute(0, 2, 3, 1)[:, :, :, None]
            featmaps_dense = F.grid_sample(featmaps, pixels_norm, align_corners=False, mode='nearest').permute(0, 2, 3, 1)[:, :, :, None]

            brdf_feature = curr_model.brdf_net(rgb_sampled, featmaps_dense, viewdir, proj_err, normal_pred, DL_target).permute(0, 3, 1, 2)
            if cfg.BRDF.refine.use:
                refine_input = torch.cat([target_rgbdc, brdf_feature], dim=1)
                if cfg.BRDF.refine.context_concat:
                    refine_input = torch.cat([refine_input, featmaps_dense.squeeze(-2).permute(0, 3, 1, 2)], dim=1)
                    brdf_refined = curr_model.brdf_refine_net(refine_input)
                else:
                    brdf_refined = curr_model.brdf_refine_net(refine_input)

            segBRDF = data['mask'][:, :1]
            albedo_pred_refined = brdf_refined[0]
            rough_pred_refined = brdf_refined[1]
            if torch.sum(torch.isnan(rough_pred_refined)) > 0:
                raise Exception('nan..')
            pred['rough'] = rough_pred_refined
            pred['conf'] = torch.ones_like(rough_pred_refined)

            albedo_pred_scaled_refined = LSregress(albedo_pred_refined.detach() * segBRDF, data['albedo_gt'] * segBRDF, albedo_pred_refined)
            albedo_pred_scaled_refined = torch.clamp(albedo_pred_scaled_refined, 0, 1)
            pred['albedo'] = albedo_pred_scaled_refined

            albedo_mse_err_refined = img2mse(albedo_pred_scaled_refined, data['albedo_gt'], segBRDF, None)
            rough_mse_err_refined = img2mse(rough_pred_refined, data['rough_gt'], segBRDF, None)

            scalars_to_log['train/albedo_mse_err'] = albedo_mse_err_refined.item()
            scalars_to_log['train/rough_mse_err'] = rough_mse_err_refined.item()
            total_loss = cfg.BRDF.lambda_albedo * albedo_mse_err_refined + cfg.BRDF.lambda_rough * rough_mse_err_refined
            scalars_to_log['train/total_loss'] = total_loss.item()
        else:
            raise Exception('stage error')

    return total_loss, pred


def compute_projection(pixel_batch, int_list, c2w_list, depth_list, im_list):
    bn, vn, _, h, w = depth_list.shape
    w2c_list = torch.inverse(c2w_list)
    pixel_depth = depth_list[:, 0, 0, ..., None, None]

    cam_coord = pixel_depth * torch.inverse(int_list[:, None, None, 0]) @ pixel_batch
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
    # weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
    # weight = weight / (torch.sum(weight, dim=1, keepdim=True) + TINY_NUMBER)
    # for i in range(7):
    #     cv2.imwrite(f'corner_0_{i}.png', cv2fromtorch(weight[0, i]))
    #     cv2.imwrite(f'corner_1_{i}.png', cv2fromtorch(weight[1, i]))

    # print(torch.mean(torch.abs(pixel_rgbd_k[0,0, -1] - depth_list[0, 0, 0])))
    # print(torch.mean(torch.abs(pixel_rgbd_k[0, 0, :3] - im_list[0, 0, :])))

    rgb_sampled = pixel_rgbd_k[:, :, :3].permute(0, 3, 4, 1, 2)
    # for i in range(7):
    #     cv2.imwrite(f'corner_0_{i}.png', cv2fromtorch(rgb_sampled[0, i]))
    #     cv2.imwrite(f'corner_1_{i}.png', cv2fromtorch(rgb_sampled[1, i]))
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
    return rgb_sampled, viewdir.permute(0, 2, 3, 1, 4), proj_err.permute(0, 2, 3, 1, 4)


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
