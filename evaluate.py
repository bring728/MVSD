import torch
import numpy as np
import argparse
import random
import os
import models
import utils
import glob
import os.path as osp
import cv2
import torch.nn.functional as F
import scipy.io as io
import utils
from tqdm import tqdm


outfilename = 'cis_output'
os.makedirs(outfilename, exist_ok=True)

val_dataset = Openrooms_FF(dataRoot, cfg, stage, 'TRAIN', False)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=3, shuffle=False)

experiments = ['models/check_cascade0_w320_h240', 'models/check_cascade1_w320_h240']
experimentsLight = ['models/check_cascadeLight0_sg12_offset1.0', 'models/check_cascadeLight1_sg12_offset1.0']
experimentsBS = ['models/checkBs_cascade0_w320_h240', 'models/checkBs_cascade1_w320_h240']
nepochs = [14, 7]
nepochsLight = [10, 10]
nepochsBS = [15, 8]
nitersBS = [1000, 4500]

imHeights = [240, 240]
imWidths = [320, 320]

seed = 0
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

encoders = []
albedoDecoders = []
normalDecoders = []
roughDecoders = []
depthDecoders = []

lightEncoders = []
axisDecoders = []
lambDecoders = []
weightDecoders = []

albedoBSs = []
depthBSs = []
roughBSs = []

for n in range(0, 2):
    # BRDF Predictioins
    encoders.append(models.encoder0(cascadeLevel=n).eval())
    albedoDecoders.append(models.decoder0(mode=0).eval())
    normalDecoders.append(models.decoder0(mode=1).eval())
    roughDecoders.append(models.decoder0(mode=2).eval())
    depthDecoders.append(models.decoder0(mode=4).eval())

    # Load weight
    encoders[n].load_state_dict(torch.load('{0}/encoder{1}_{2}.pth'.format(experiments[n], n, nepochs[n] - 1)).state_dict())
    albedoDecoders[n].load_state_dict(torch.load('{0}/albedo{1}_{2}.pth'.format(experiments[n], n, nepochs[n] - 1)).state_dict())
    normalDecoders[n].load_state_dict(torch.load('{0}/normal{1}_{2}.pth'.format(experiments[n], n, nepochs[n] - 1)).state_dict())
    roughDecoders[n].load_state_dict(torch.load('{0}/rough{1}_{2}.pth'.format(experiments[n], n, nepochs[n] - 1)).state_dict())
    depthDecoders[n].load_state_dict(torch.load('{0}/depth{1}_{2}.pth'.format(experiments[n], n, nepochs[n] - 1)).state_dict())

    for param in encoders[n].parameters():
        param.requires_grad = False
    for param in albedoDecoders[n].parameters():
        param.requires_grad = False
    for param in normalDecoders[n].parameters():
        param.requires_grad = False
    for param in roughDecoders[n].parameters():
        param.requires_grad = False
    for param in depthDecoders[n].parameters():
        param.requires_grad = False

    lightEncoders.append(models.encoderLight(cascadeLevel=n, SGNum=12).eval())
    axisDecoders.append(models.decoderLight(mode=0, SGNum=12).eval())
    lambDecoders.append(models.decoderLight(mode=1, SGNum=12).eval())
    weightDecoders.append(models.decoderLight(mode=2, SGNum=12).eval())

    lightEncoders[n].load_state_dict(torch.load('{0}/lightEncoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)).state_dict())
    axisDecoders[n].load_state_dict(torch.load('{0}/axisDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)).state_dict())
    lambDecoders[n].load_state_dict(torch.load('{0}/lambDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)).state_dict())
    weightDecoders[n].load_state_dict(torch.load('{0}/weightDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)).state_dict())

    for param in lightEncoders[n].parameters():
        param.requires_grad = False
    for param in axisDecoders[n].parameters():
        param.requires_grad = False
    for param in lambDecoders[n].parameters():
        param.requires_grad = False
    for param in weightDecoders[n].parameters():
        param.requires_grad = False

    albedoBSs.append(bs.BilateralLayer(mode=0))
    roughBSs.append(bs.BilateralLayer(mode=2))
    depthBSs.append(bs.BilateralLayer(mode=4))

    albedoBSs[n].load_state_dict(torch.load('{0}/albedoBs{1}_{2}_{3}.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])).state_dict())
    roughBSs[n].load_state_dict(torch.load('{0}/roughBs{1}_{2}_{3}.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])).state_dict())
    depthBSs[n].load_state_dict(torch.load('{0}/depthBs{1}_{2}_{3}.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])).state_dict())

    for param in albedoBSs[n].parameters():
        param.requires_grad = False
    for param in roughBSs[n].parameters():
        param.requires_grad = False
    for param in depthBSs[n].parameters():
        param.requires_grad = False

for n in range(0, 2):
    encoders[n] = encoders[n].cuda()
    albedoDecoders[n] = albedoDecoders[n].cuda()
    normalDecoders[n] = normalDecoders[n].cuda()
    roughDecoders[n] = roughDecoders[n].cuda()
    depthDecoders[n] = depthDecoders[n].cuda()

    albedoBSs[n] = albedoBSs[n].cuda()
    roughBSs[n] = roughBSs[n].cuda()
    depthBSs[n] = depthBSs[n].cuda()

    lightEncoders[n] = lightEncoders[n].cuda()
    axisDecoders[n] = axisDecoders[n].cuda()
    lambDecoders[n] = lambDecoders[n].cuda()
    weightDecoders[n] = weightDecoders[n].cuda()


j = 0
for data in tqdm(val_loader):
    j += 1
    imBatches = []
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    newImWidth = []
    newImHeight = []
    for n in range(2):
        if nh < nw:
            newW = imWidths[n]
            newH = int(float(imWidths[n]) / float(nw) * nh)
        else:
            newH = imHeights[n]
            newW = int(float(imHeights[n]) / float(nh) * nw)

        if nh < newH:
            im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_AREA)
            imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                         align_corners=True)

        else:
            im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_LINEAR)
            imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                         align_corners=True)

        newImWidth.append(newW)
        newImHeight.append(newH)

        im = (np.transpose(im, [2, 0, 1]).astype(np.float32) / 255.0)[np.newaxis, :, :, :]
        im = im / im.max()
        imBatches.append(torch.from_numpy(im ** (2.2)).cuda())

    nh, nw = newImHeight[-1], newImWidth[-1]

    newEnvWidth, newEnvHeight, fov = 0, 0, 0
    if nh < nw:
        fov = 57
        newW = 160
        newH = int(float(160) / float(nw) * nh)
    else:
        fov = 42.75
        newH = 120
        newW = int(float(120) / float(nh) * nw)

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_AREA)
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_LINEAR)

    newEnvWidth = newW
    newEnvHeight = newH

    im = (np.transpose(im, [2, 0, 1]).astype(np.float32) / 255.0)[np.newaxis, :, :, :]
    im = im / im.max()
    imBatchSmall = torch.from_numpy(im ** (2.2)).cuda()
    renderLayer = models.renderingLayer(imWidth=newEnvWidth, imHeight=newEnvHeight, fov=fov, envWidth=16, envHeight=8)

    output2env = models.output2env(envWidth=16, envHeight=8, SGNum=12)

    ########################################################
    # Build the cascade network architecture #
    albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
    albedoBSPreds, roughBSPreds, depthBSPreds = [], [], []
    envmapsPreds, envmapsPredImages, renderedPreds = [], [], []
    cAlbedos = []
    cLights = []

    ################# BRDF Prediction ######################
    inputBatch = imBatches[0]
    x1, x2, x3, x4, x5, x6 = encoders[0](inputBatch)

    albedoPred = 0.5 * (albedoDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (depthDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)

    # Normalize Albedo and depth
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)

    bn, ch, nrow, ncol = depthPred.size()
    depthPred = depthPred.view(bn, -1)
    depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    depthPred = depthPred.view(bn, ch, nrow, ncol)

    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)

    imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear', align_corners=True)
    albedoPredLarge = F.interpolate(albedoPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear', align_corners=True)
    normalPredLarge = F.interpolate(normalPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear', align_corners=True)
    roughPredLarge = F.interpolate(roughPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear', align_corners=True)
    depthPredLarge = F.interpolate(depthPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear', align_corners=True)

    inputBatch = torch.cat([imBatchLarge, albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge], dim=1)
    x1, x2, x3, x4, x5, x6 = lightEncoders[0](inputBatch)

    # Prediction
    axisPred = axisDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
    lambPred = lambDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
    weightPred = weightDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
    bn, SGNum, _, envRow, envCol = axisPred.size()
    envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred], dim=1)
    envmapsPreds.append(envmapsPred)

    envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred)
    envmapsPredImages.append(envmapsPredImage)

    diffusePred, specularPred = renderLayer.forwardEnv(albedoPreds[0], normalPreds[0], roughPreds[0], envmapsPredImages[0])

    diffusePredNew, specularPredNew = models.LSregressDiffSpec(diffusePred, specularPred, imBatchSmall, diffusePred, specularPred)
    renderedPred = diffusePredNew + specularPredNew
    renderedPreds.append(renderedPred)

    cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredNew)) / (torch.sum(specularPred))).data.item()
    if cSpec < 1e-3:
        cAlbedo = 1 / albedoPreds[-1].max().data.item()
        cLight = cDiff / cAlbedo
    else:
        cLight = cSpec
        cAlbedo = cDiff / cLight
        cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item())
        cLight = cDiff / cAlbedo
    envmapsPredImages[0] = envmapsPredImages[0] * cLight
    cAlbedos.append(cAlbedo)
    cLights.append(cLight)

    diffusePred = diffusePredNew
    specularPred = specularPredNew

    albedoPredLarge = F.interpolate(albedoPreds[0], [newImHeight[1], newImWidth[1]], mode='bilinear')
    normalPredLarge = F.interpolate(normalPreds[0], [newImHeight[1], newImWidth[1]], mode='bilinear')
    roughPredLarge = F.interpolate(roughPreds[0], [newImHeight[1], newImWidth[1]], mode='bilinear')
    depthPredLarge = F.interpolate(depthPreds[0], [newImHeight[1], newImWidth[1]], mode='bilinear')

    diffusePredLarge = F.interpolate(diffusePred, [newImHeight[1], newImWidth[1]], mode='bilinear')
    specularPredLarge = F.interpolate(specularPred, [newImHeight[1], newImWidth[1]], mode='bilinear')

    inputBatch = torch.cat([imBatches[1], albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge, diffusePredLarge, specularPredLarge], dim=1)

    x1, x2, x3, x4, x5, x6 = encoders[1](inputBatch)
    albedoPred = 0.5 * (albedoDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (depthDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)

    # Normalize Albedo and depth
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)

    bn, ch, nrow, ncol = depthPred.size()
    depthPred = depthPred.view(bn, -1)
    depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    depthPred = depthPred.view(bn, ch, nrow, ncol)

    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)

    imBatchLarge = F.interpolate(imBatches[1], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear')
    albedoPredLarge = F.interpolate(albedoPreds[1], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear')
    normalPredLarge = F.interpolate(normalPreds[1], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear')
    roughPredLarge = F.interpolate(roughPreds[1], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear')
    depthPredLarge = F.interpolate(depthPreds[1], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear')

    inputBatch = torch.cat([imBatchLarge, albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge], dim=1)
    x1, x2, x3, x4, x5, x6 = lightEncoders[1](inputBatch, envmapsPred)

    # Prediction
    axisPred = axisDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
    lambPred = lambDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
    weightPred = weightDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
    bn, SGNum, _, envRow, envCol = axisPred.size()
    envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred], dim=1)
    envmapsPreds.append(envmapsPred)

    envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred)
    envmapsPredImages.append(envmapsPredImage)

    diffusePred, specularPred = renderLayer.forwardEnv(albedoPreds[1], normalPreds[1], roughPreds[1], envmapsPredImages[1])

    diffusePredNew, specularPredNew = models.LSregressDiffSpec(diffusePred, specularPred, imBatchSmall, diffusePred, specularPred)

    renderedPre = diffusePredNew + specularPredNew
    renderedPreds.append(renderedPred)

    cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredNew)) / (torch.sum(specularPred))).data.item()
    if cSpec == 0:
        cAlbedo = 1 / albedoPreds[-1].max().data.item()
        cLight = cDiff / cAlbedo
    else:
        cLight = cSpec
        cAlbedo = cDiff / cLight
        cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item())
        cLight = cDiff / cAlbedo
    envmapsPredImages[-1] = envmapsPredImages[-1] * cLight
    cAlbedos.append(cAlbedo)
    cLights.append(cLight)

    diffusePred = diffusePredNew
    specularPred = specularPredNew

    for n in range(0, 2):
        albedoBSPred, albedoConf = albedoBSs[n](imBatches[n], albedoPreds[n].detach(), albedoPreds[n])
        albedoBSPreds.append(albedoBSPred)
        roughBSPred, roughConf = roughBSs[n](imBatches[n], albedoPreds[n].detach(), 0.5 * (roughPreds[n] + 1))
        roughBSPred = torch.clamp(2 * roughBSPred - 1, -1, 1)
        roughBSPreds.append(roughBSPred)
        depthBSPred, depthConf = depthBSs[n](imBatches[n], albedoPreds[n].detach(), depthPreds[n])
        depthBSPreds.append(depthBSPred)

    for n in range(0, len(albedoPreds)):
        if n < len(cAlbedos):
            albedoPred = (albedoPreds[n] * cAlbedos[n]).data.cpu().numpy().squeeze()
        else:
            albedoPred = albedoPreds[n].data.cpu().numpy().squeeze()

        albedoPred = albedoPred.transpose([1, 2, 0])
        albedoPred = (albedoPred) ** (1.0 / 2.2)
        albedoPred = cv2.resize(albedoPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

        albedoPredIm = (np.clip(255 * albedoPred, 0, 255)).astype(np.uint8)

        cv2.imwrite(albedoImNames[n], albedoPredIm[:, :, ::-1])

    # Save the normal
    for n in range(0, len(normalPreds)):
        normalPred = normalPreds[n].data.cpu().numpy().squeeze()
        normalPred = normalPred.transpose([1, 2, 0])
        normalPred = cv2.resize(normalPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

        np.save(normalNames[n], normalPred)

        normalPredIm = (255 * 0.5 * (normalPred + 1)).astype(np.uint8)
        cv2.imwrite(normalImNames[n], normalPredIm[:, :, ::-1])

    # Save the rough
    for n in range(0, len(roughPreds)):
        roughPred = roughPreds[n].data.cpu().numpy().squeeze()
        roughPred = cv2.resize(roughPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

        roughPredIm = (255 * 0.5 * (roughPred + 1)).astype(np.uint8)
        cv2.imwrite(roughImNames[n], roughPredIm)

    # Save the depth
    for n in range(0, len(depthPreds)):
        depthPred = depthPreds[n].data.cpu().numpy().squeeze()
        np.save(depthNames[n], depthPred)

        depthPred = depthPred / np.maximum(depthPred.mean(), 1e-10) * 3
        depthPred = cv2.resize(depthPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

        depthOut = 1 / np.clip(depthPred + 1, 1e-6, 10)
        depthPredIm = (255 * depthOut).astype(np.uint8)
        cv2.imwrite(depthImNames[n], depthPredIm)

    for n in range(0, len(albedoBSPreds)):
        if n < len(cAlbedos):
            albedoBSPred = (albedoBSPreds[n] * cAlbedos[n]).data.cpu().numpy().squeeze()
        else:
            albedoBSPred = albedoBSPreds[n].data.cpu().numpy().squeeze()
        albedoBSPred = albedoBSPred.transpose([1, 2, 0])
        albedoBSPred = (albedoBSPred) ** (1.0 / 2.2)
        albedoBSPred = cv2.resize(albedoBSPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

        albedoBSPredIm = (np.clip(255 * albedoBSPred, 0, 255)).astype(np.uint8)
        cv2.imwrite(albedoImNames[n].replace('albedo', 'albedoBS'), albedoBSPredIm[:, :, ::-1])

        # Save the rough bs
        for n in range(0, len(roughBSPreds)):
            roughBSPred = roughBSPreds[n].data.cpu().numpy().squeeze()
            roughBSPred = cv2.resize(roughBSPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

            roughBSPredIm = (255 * 0.5 * (roughBSPred + 1)).astype(np.uint8)
            cv2.imwrite(roughImNames[n].replace('rough', 'roughBS'), roughBSPredIm)

        for n in range(0, len(depthBSPreds)):
            depthBSPred = depthBSPreds[n].data.cpu().numpy().squeeze()
            np.save(depthNames[n].replace('depth', 'depthBS'), depthBSPred)

            depthBSPred = depthBSPred / np.maximum(depthBSPred.mean(), 1e-10) * 3
            depthBSPred = cv2.resize(depthBSPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

            depthOut = 1 / np.clip(depthBSPred + 1, 1e-6, 10)
            depthBSPredIm = (255 * depthOut).astype(np.uint8)
            cv2.imwrite(depthImNames[n].replace('depth', 'depthBS'), depthBSPredIm)

    # Save the envmapImages
    for n in range(0, len(envmapsPredImages)):
        envmapsPredImage = envmapsPredImages[n].data.cpu().numpy().squeeze()
        envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])

        # Flip to be conincide with our dataset
        # np.savez_compressed(envmapPredImNames[n], env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
        np.savez(envmapPredImNames[n], env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
        utils.writeEnvToFile(envmapsPredImages[n], 0, envmapPredImNames[n], nrows=24, ncols=16)

        for n in range(0, len(envmapsPreds)):
            envmapsPred = envmapsPreds[n].data.cpu().numpy()
            np.save(envmapsPredSGNames[n], envmapsPred)
            shading = utils.predToShading(envmapsPred, SGNum=12)
            shading = shading.transpose([1, 2, 0])
            shading = shading / np.mean(shading) / 3.0
            shading = np.clip(shading, 0, 1)
            shading = (255 * shading ** (1.0 / 2.2)).astype(np.uint8)
            cv2.imwrite(shadingNames[n], shading[:, :, ::-1])

        for n in range(0, len(cLights)):
            io.savemat(cLightNames[n], {'cLight': cLights[n]})

        # Save the rendered image
        for n in range(0, len(renderedPreds)):
            renderedPred = renderedPreds[n].data.cpu().numpy().squeeze()
            renderedPred = renderedPred.transpose([1, 2, 0])
            renderedPred = (renderedPred / renderedPred.max()) ** (1.0 / 2.2)
            renderedPred = cv2.resize(renderedPred, (nw, nh), interpolation=cv2.INTER_LINEAR)
            # np.save(renderedNames[n], renderedPred )

            renderedPred = (np.clip(renderedPred, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(renderedImNames[n], renderedPred[:, :, ::-1])

    # Save the image
    cv2.imwrite(imOutputNames[0], im_cpu[:, :, ::-1])
