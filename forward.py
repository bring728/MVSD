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
    # data['depth_norm'] = data['depth_norm'][:, :num_view]
    # data['conf'] = data['conf'][:, :num_view]
    data['cam'] = data['cam'][:, :num_view]
    data['c2w'] = data['c2w'][:, :num_view]


def model_forward(stage, phase, curr_model, helper_dict, data, cfg, scalars_to_log, save_image_flag):
    with autocast(enabled=cfg.autocast):
        pred = {}
        total_loss = None
        if stage == '1':
            if cfg.mode == 'normal':
                pred['normal'] = curr_model.normal_net(data['input'])
                normal_mse_err = img2mse(pred['normal'], data['normal_gt'], data['mask'][:, 1:, ...])
                normal_ang_err = img2angerr(pred['normal'], data['normal_gt'], data['mask'][:, 1:, ...])
                scalars_to_log['train/normal_mse_err'] = normal_mse_err.item()
                scalars_to_log['train/normal_ang_err'] = normal_ang_err.item()
                total_loss = cfg.normal.lambda_mse * normal_mse_err + cfg.normal.lambda_ang * normal_ang_err
                scalars_to_log['train/total_loss'] = total_loss.item()
            elif cfg.mode == 'DL':
                with torch.no_grad():
                    pred['normal'] = curr_model.normal_net(data['input'])
                rgbdcn = torch.cat([data['input'][:, :5], 0.5 * (pred['normal'] + 1)], dim=1)
                axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
                sharpness_hdr, intensity_hdr = helper_dict['sg2env'].SG_ldr2hdr(sharpness, intensity)
                envmaps_DL_pred = helper_dict['sg2env'].fromSGtoIm(axis, sharpness_hdr, intensity_hdr)
                pred['envmaps_DL'] = envmaps_DL_pred

                segBRDF = F.adaptive_avg_pool2d(data['mask'][:, :1, ...], (cfg.DL.env_rows, cfg.DL.env_cols))
                # notDarkEnv = (torch.mean(data['envmaps_DL_gt'], dim=(1, 4, 5)) > 0.001).float()[:, None]
                # segEnvBatch = (segBRDF * notDarkEnv)[..., None, None].expand_as(data['envmaps_DL_gt'])
                segEnvBatch = segBRDF[..., None, None].expand_as(data['envmaps_DL_gt'])
                if cfg.DL.scale_inv:
                    envmaps_DL_pred_scaled = LSregress(envmaps_DL_pred.detach() * segEnvBatch, data['envmaps_DL_gt'] * segEnvBatch, envmaps_DL_pred)
                    env_loss = img2log_mse(envmaps_DL_pred_scaled, data['envmaps_DL_gt'], segEnvBatch)
                    scalars_to_log['train/env_scaled_loss'] = env_loss.item()
                else:
                    env_loss = img2log_mse(envmaps_DL_pred, data['envmaps_DL_gt'], segEnvBatch)
                    scalars_to_log['train/env_msle_loss'] = env_loss.item()

                vis_beta_loss = torch.mean(torch.log(0.1 + vis) + torch.log(0.1 + 1. - vis) + 2.20727)  # from neural volumes
                # vis_beta_loss = torch.mean(torch.log10(0.1 + vis) + torch.log10(0.1 + 1. - vis) + 1.)
                scalars_to_log['train/vis_beta_loss'] = vis_beta_loss.item()
                total_loss = env_loss + cfg.DL.lambda_vis_prior * vis_beta_loss

            # if cfg.mode == 'finetune':
            #     pred['normal'] = curr_model.normal_net(data['input'])
            #     normal_mse_err = img2mse(pred['normal'], data['normal_gt'], data['mask'][:, 1:, ...])
            #     normal_ang_err = img2angerr(pred['normal'], data['normal_gt'], data['mask'][:, 1:, ...])
            #     scalars_to_log['train/normal_mse_err'] = normal_mse_err.item()
            #     scalars_to_log['train/normal_ang_err'] = normal_ang_err.item()
            #
            #     rgbdcn = torch.cat([data['input'], 0.5 * (pred['normal'] + 1)], dim=1)
            #     axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
            #     sharpness_hdr, intensity_hdr = helper_dict['sg2env'].SG_ldr2hdr(sharpness, intensity)
            #     envmaps_DL_pred = helper_dict['sg2env'].fromSGtoIm(axis, sharpness_hdr, intensity_hdr)
            #     pred['envmaps_DL'] = envmaps_DL_pred
            #
            #     segBRDF = F.adaptive_avg_pool2d(data['mask'][:, :1, ...], (cfg.DL.env_rows, cfg.DL.env_cols))
            #     segEnvBatch = segBRDF[..., None, None].expand_as(data['envmaps_DL_gt'])
            #     if cfg.DL.scale_inv:
            #         envmaps_DL_pred_scaled = LSregress(envmaps_DL_pred.detach() * segEnvBatch, data['envmaps_DL_gt'] * segEnvBatch, envmaps_DL_pred)
            #         env_loss = img2log_mse(envmaps_DL_pred_scaled, data['envmaps_DL_gt'], segEnvBatch)
            #         scalars_to_log['train/env_scaled_loss'] = env_loss.item()
            #     else:
            #         env_loss = img2log_mse(envmaps_DL_pred, data['envmaps_DL_gt'], segEnvBatch)
            #         scalars_to_log['train/env_msle_loss'] = env_loss.item()
            #     vis_beta_loss = torch.mean(torch.log(0.1 + vis) + torch.log(0.1 + 1. - vis) + 2.20727)  # from neural volumes
            #     scalars_to_log['train/vis_beta_loss'] = vis_beta_loss.item()
            #     total_loss = env_loss + cfg.lambda_mse * normal_mse_err + cfg.lambda_ang * normal_ang_err + cfg.lambda_vis_prior * vis_beta_loss
            #     scalars_to_log['train/total_loss'] = total_loss.item()

        elif stage == '2':
            # sample_view(data, 7 + np.random.choice(3, 1)[0], gt=cfg.BRDF.gt)
            if 'grad' in data:
                normal_input = torch.cat([data['rgb'][:, 0], data['depth_norm'], data['conf'], data['grad']], dim=1)
            else:
                normal_input = torch.cat([data['rgb'][:, 0], data['depth_norm'], data['conf']], dim=1)
            with torch.no_grad():
                data['normal'] = curr_model.normal_net(normal_input)
                rgbdcn = torch.cat([normal_input[:, :5], 0.5 * (data['normal'] + 1)], dim=1)
                axis, sharpness, intensity, vis = curr_model.DL_net(rgbdcn)
                sharpness, intensity = helper_dict['sg2env'].SG_ldr2hdr(sharpness, intensity)
                bn, _, _, DL_rows, DL_cols = axis.shape
            if save_image_flag:
                data['envmaps_DL'] = helper_dict['sg2env'].fromSGtoIm(axis, sharpness, intensity)

            pixels = helper_dict['pixels']
            up = helper_dict['up']
            if helper_dict['pixels_norm'].size(0) != bn:
                pixels_norm = helper_dict['pixels_norm'][:bn]
            else:
                pixels_norm = helper_dict['pixels_norm']

            x_encoded, featmaps = curr_model.context_net(rgbdcn)
            if cfg.BRDF.gt:
                rgb_sampled, viewdir, proj_err = compute_projection(pixels, data['cam'], data['c2w'], data['depth_gt'], data['rgb'])
            else:
                rgb_sampled, viewdir, proj_err = compute_projection(pixels, data['cam'], data['c2w'], data['depth_est'], data['rgb'])
            normal_pred = data['normal'].permute(0, 2, 3, 1)[:, :, :, None, None]

            axis = normal2camera(data['normal'], axis, up)
            data['DL'] = torch.cat([axis, sharpness, intensity], dim=2).reshape((bn, -1, DL_rows, DL_cols))
            DL_target = F.grid_sample(data['DL'], pixels_norm, align_corners=False, mode='nearest').permute(0, 2, 3, 1)[:, :, :, None]
            featmaps_dense = F.grid_sample(featmaps, pixels_norm, align_corners=False, mode='bilinear').permute(0, 2, 3, 1)[:, :, :, None]

            brdf_feature = curr_model.aggregation_net(rgb_sampled, featmaps_dense, viewdir.unsqueeze(-2), proj_err, normal_pred, DL_target).permute(0, 3, 1, 2)
            refine_input = torch.cat([rgbdcn, brdf_feature, featmaps_dense.squeeze(-2).permute(0, 3, 1, 2)], dim=1)
            albedo, rough = curr_model.brdf_refine_net(refine_input)

            segBRDF = data['mask'][:, :1]
            pred['rough'] = rough
            pred['conf'] = torch.ones_like(rough)

            albedo_pred_scaled_refined = torch.clamp(LSregress(albedo.detach() * segBRDF, data['albedo_gt'] * segBRDF, albedo), 0, 1)
            pred['albedo'] = albedo_pred_scaled_refined

            albedo_mse_err_refined = img2mse(albedo_pred_scaled_refined, data['albedo_gt'], segBRDF, None)
            rough_mse_err_refined = img2mse(rough, data['rough_gt'], segBRDF, None)

            scalars_to_log['train/albedo_mse_err'] = albedo_mse_err_refined.item()
            scalars_to_log['train/rough_mse_err'] = rough_mse_err_refined.item()
            total_loss = cfg.BRDF.lambda_albedo * albedo_mse_err_refined + cfg.BRDF.lambda_rough * rough_mse_err_refined

            if cfg.mode == 'finetune':
                if helper_dict['voxel_grid_front'].size(0) != bn:
                    voxel_grid_front = helper_dict['voxel_grid_front'][:bn]
                    # voxel_grid = torch.cat([helper_dict['voxel_grid_back'][:bn], voxel_grid_front], dim=-2)
                    ls = helper_dict['ls'][:bn]
                else:
                    voxel_grid_front = helper_dict['voxel_grid_front']
                    # voxel_grid = torch.cat([helper_dict['voxel_grid_back'], voxel_grid_front], dim=-2)
                    ls = helper_dict['ls']

                global_feature_volume = curr_model.GL_Net(x_encoded)

                source = torch.cat([rgbdcn, albedo, rough, brdf_feature], dim=1)
                visible_surface_volume = get_visible_surface_volume(voxel_grid_front, source, data['cam'][:, 0])
                VSG = curr_model.VSG_Net(visible_surface_volume, global_feature_volume)

                envmaps_SVL = envmap_from_VSG(VSG, ls, up, pixels, data['cam'][:, 0], data['depth_norm'], normal_pred)

            scalars_to_log['train/total_loss'] = total_loss.item()

        else:
            raise Exception('stage error')

    return total_loss, pred


def get_visible_surface_volume(voxel_grid, source, intrinsic):
    bn, c, h, w = source.shape
    bn, c1, c2, c3, _ = voxel_grid.shape

    # get projection coord
    pixel_coord = (intrinsic[:, None, None, None] @ voxel_grid[..., None])[..., 0]
    if torch.min(pixel_coord[..., 2:3]) < 0:
        raise Exception('projection error')
    pixel_coord = pixel_coord[..., :2] / pixel_coord[..., 2:3]

    resize_factor = torch.tensor([w, h]).to(pixel_coord.device)[None, None, None, None, :]
    pixel_coord_norm = (2 * (pixel_coord / resize_factor) - 1.).reshape(bn, c1, c2 * c3, 2)
    unprojected_volume = F.grid_sample(source, pixel_coord_norm, align_corners=False, mode='bilinear').reshape(bn, c, c1, c2, c3)
    visible_surface_volume = torch.cat([unprojected_volume[:, :3], unprojected_volume[:, 5:]], dim=1)
    volume_weight_k = torch.exp(unprojected_volume[:, 4, ...] * -torch.pow(unprojected_volume[:, 3, ...] - voxel_grid[..., -1], 2))
    return visible_surface_volume * volume_weight_k.unsqueeze(1)


def envmap_from_VSG(VSG, ls, up, pixel_batch, intrinsic, depth, normal):
    pixel_batch = pixel_batch[:, ::2, ::2]
    cam_coord = (depth[:, 0, ::2, ::2, None, None] * (torch.inverse(intrinsic[:, None, None]) @ pixel_batch)).squeeze(-1)
    normal = normal[:, ::2, ::2, 0]
    camyProj = torch.einsum('d,abcd->abc', (up, normal)).unsqueeze(-1) * normal
    camy = F.normalize(up[None, None, None] - camyProj, dim=-1)
    camx = -F.normalize(torch.cross(camy, normal, dim=-1), dim=-1)
    axis = ls[..., 0:1] * camx[:, :, :, None] + ls[..., 0:1] * camy[:, :, :, None] + ls[..., 0:1] * normal[:, :, :, None] + cam_coord[:, :, :, None]
    VSG


def normal2camera(normal, axis_org, up):
    normal = F.adaptive_avg_pool2d(normal, (axis_org.size(-2), axis_org.size(-1)))
    camyProj = torch.einsum('b,abcd->acd', (up, normal)).unsqueeze(1) * normal
    camy = F.normalize(up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - camyProj, dim=1)
    camx = -F.normalize(torch.cross(camy, normal, dim=1), dim=1)
    axis = axis_org[:, :, 0:1] * camx.unsqueeze(1) + axis_org[:, :, 1:2] * camy.unsqueeze(1) + axis_org[:, :, 2:3] * normal.unsqueeze(1)
    return axis


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
    pixel_coord_k_norm = (2.0 * (pixel_coord_k_est / resize_factor) - 1.).reshape([bn * vn, h, w, 2])
    pixel_rgbd_k = F.grid_sample(torch.cat([im_list, depth_list], dim=2).reshape([bn * vn, 4, h, w]), pixel_coord_k_norm,
                                 align_corners=False, mode='bilinear').reshape([bn, vn, 4, h, w])
    proj_err = pixel_depth_k_est - pixel_rgbd_k[:, :, -1, ..., None]
    rgb_sampled = pixel_rgbd_k[:, :, :3].permute(0, 3, 4, 1, 2)

    viewdir = F.normalize((c2w_list[:, :, None, None, :3, :3] @ cam_coord_k)[..., 0], dim=-1)
    viewdir[..., :1] = -viewdir[..., :1]  # because openrooms is using right, up, backward coordinates and view dir is pixel -> camera
    return rgb_sampled, viewdir.permute(0, 2, 3, 1, 4), proj_err.permute(0, 2, 3, 1, 4)



    # weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
    # weight = weight / (torch.sum(weight, dim=1, keepdim=True) + TINY_NUMBER)
    # for i in range(7):
    #     cv2.imwrite(f'corner_0_{i}.png', cv2fromtorch(weight[0, i]))
    #     cv2.imwrite(f'corner_1_{i}.png', cv2fromtorch(weight[1, i]))

    # print(torch.mean(torch.abs(pixel_rgbd_k[0,0, -1] - depth_list[0, 0, 0])))
    # print(torch.mean(torch.abs(pixel_rgbd_k[0, 0, :3] - im_list[0, 0, :])))
    # for i in range(7):
    #     cv2.imwrite(f'corner_0_{i}.png', cv2fromtorch(rgb_sampled[0, i]))
    #     cv2.imwrite(f'corner_1_{i}.png', cv2fromtorch(rgb_sampled[1, i]))
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
