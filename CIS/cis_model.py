import torch
import numpy as np
from CIS import cis_network
import cv2
from CIS import BilateralLayer as bs
import torch.nn.functional as F


class CIS_model(object):
    def __init__(self):
        experiments = ['CIS/cis_network_weights/check_cascade0_w320_h240', 'CIS/cis_network_weights/check_cascade1_w320_h240']
        experimentsLight = ['CIS/cis_network_weights/check_cascadeLight0_sg12_offset1.0', 'CIS/cis_network_weights/check_cascadeLight1_sg12_offset1.0']
        experimentsBS = ['CIS/cis_network_weights/checkBs_cascade0_w320_h240', 'CIS/cis_network_weights/checkBs_cascade1_w320_h240']
        nepochs = [14, 7]
        nepochsLight = [10, 10]
        nepochsBS = [15, 8]
        nitersBS = [1000, 4500]

        self.imHeights = [480, 480]
        self.imWidths = [640, 640]

        self.encoders = []
        self.albedoDecoders = []
        self.normalDecoders = []
        self.roughDecoders = []
        self.depthDecoders = []

        self.lightEncoders = []
        self.axisDecoders = []
        self.lambDecoders = []
        self.weightDecoders = []

        self.albedoBSs = []
        self.depthBSs = []
        self.roughBSs = []

        for n in range(0, 2):
            # BRDF Predictioins
            self.encoders.append(cis_network.encoder0(cascadeLevel=n).eval())
            self.albedoDecoders.append(cis_network.decoder0(mode=0).eval())
            self.normalDecoders.append(cis_network.decoder0(mode=1).eval())
            self.roughDecoders.append(cis_network.decoder0(mode=2).eval())
            self.depthDecoders.append(cis_network.decoder0(mode=4).eval())

            # Load weight
            self.encoders[n].load_state_dict(torch.load('{0}/encoder{1}_{2}_new.pth'.format(experiments[n], n, nepochs[n] - 1)))
            self.albedoDecoders[n].load_state_dict(torch.load('{0}/albedo{1}_{2}_new.pth'.format(experiments[n], n, nepochs[n] - 1)))
            self.normalDecoders[n].load_state_dict(torch.load('{0}/normal{1}_{2}_new.pth'.format(experiments[n], n, nepochs[n] - 1)))
            self.roughDecoders[n].load_state_dict(torch.load('{0}/rough{1}_{2}_new.pth'.format(experiments[n], n, nepochs[n] - 1)))
            self.depthDecoders[n].load_state_dict(torch.load('{0}/depth{1}_{2}_new.pth'.format(experiments[n], n, nepochs[n] - 1)))

            for param in self.encoders[n].parameters():
                param.requires_grad = False
            for param in self.albedoDecoders[n].parameters():
                param.requires_grad = False
            for param in self.normalDecoders[n].parameters():
                param.requires_grad = False
            for param in self.roughDecoders[n].parameters():
                param.requires_grad = False
            for param in self.depthDecoders[n].parameters():
                param.requires_grad = False

            self.lightEncoders.append(cis_network.encoderLight(cascadeLevel=n, SGNum=12).eval())
            self.axisDecoders.append(cis_network.decoderLight(mode=0, SGNum=12).eval())
            self.lambDecoders.append(cis_network.decoderLight(mode=1, SGNum=12).eval())
            self.weightDecoders.append(cis_network.decoderLight(mode=2, SGNum=12).eval())

            self.lightEncoders[n].load_state_dict(torch.load('{0}/lightEncoder{1}_{2}_new.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)))
            self.axisDecoders[n].load_state_dict(torch.load('{0}/axisDecoder{1}_{2}_new.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)))
            self.lambDecoders[n].load_state_dict(torch.load('{0}/lambDecoder{1}_{2}_new.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)))
            self.weightDecoders[n].load_state_dict(torch.load('{0}/weightDecoder{1}_{2}_new.pth'.format(experimentsLight[n], n, nepochsLight[n] - 1)))

            for param in self.lightEncoders[n].parameters():
                param.requires_grad = False
            for param in self.axisDecoders[n].parameters():
                param.requires_grad = False
            for param in self.lambDecoders[n].parameters():
                param.requires_grad = False
            for param in self.weightDecoders[n].parameters():
                param.requires_grad = False

            self.albedoBSs.append(bs.BilateralLayer(mode=0))
            self.roughBSs.append(bs.BilateralLayer(mode=2))
            self.depthBSs.append(bs.BilateralLayer(mode=4))

            self.albedoBSs[n].load_state_dict(torch.load('{0}/albedoBs{1}_{2}_{3}_new.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])))
            self.roughBSs[n].load_state_dict(torch.load('{0}/roughBs{1}_{2}_{3}_new.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])))
            self.depthBSs[n].load_state_dict(torch.load('{0}/depthBs{1}_{2}_{3}_new.pth'.format(experimentsBS[n], n, nepochsBS[n] - 1, nitersBS[n])))

            for param in self.albedoBSs[n].parameters():
                param.requires_grad = False
            for param in self.roughBSs[n].parameters():
                param.requires_grad = False
            for param in self.depthBSs[n].parameters():
                param.requires_grad = False

        for n in range(0, 2):
            self.encoders[n] = self.encoders[n].cuda()
            self.albedoDecoders[n] = self.albedoDecoders[n].cuda()
            self.normalDecoders[n] = self.normalDecoders[n].cuda()
            self.roughDecoders[n] = self.roughDecoders[n].cuda()
            self.depthDecoders[n] = self.depthDecoders[n].cuda()

            self.albedoBSs[n] = self.albedoBSs[n].cuda()
            self.roughBSs[n] = self.roughBSs[n].cuda()
            self.depthBSs[n] = self.depthBSs[n].cuda()

            self.lightEncoders[n] = self.lightEncoders[n].cuda()
            self.axisDecoders[n] = self.axisDecoders[n].cuda()
            self.lambDecoders[n] = self.lambDecoders[n].cuda()
            self.weightDecoders[n] = self.weightDecoders[n].cuda()

        self.output2env = cis_network.output2env(envWidth=16, envHeight=8, SGNum=12)
        self.renderLayer = cis_network.renderingLayer(imWidth=160, imHeight=120, fov=57, envWidth=16, envHeight=8)


    def forward(self, im_cpu):
        imBatches = []
        nh, nw = im_cpu.shape[0], im_cpu.shape[1]

        # Resize Input Images
        newImWidth = []
        newImHeight = []
        for n in range(2):
            if nh < nw:
                newW = self.imWidths[n]
                newH = int(float(self.imWidths[n]) / float(nw) * nh)
            else:
                newH = self.imHeights[n]
                newW = int(float(self.imHeights[n]) / float(nh) * nw)

            im = im_cpu
            if nw != newW:
                if nh < newH:
                    im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_AREA) # enlarge
                else:
                    im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_LINEAR) # minify

            newImWidth.append(newW)
            newImHeight.append(newH)

            im = np.transpose(im, [2, 0, 1])[np.newaxis, :, :, :]
            imBatches.append(torch.from_numpy(im).cuda())

        nh, nw = newImHeight[-1], newImWidth[-1]
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

        im = np.transpose(im, [2, 0, 1])[np.newaxis, :, :, :]
        imBatchSmall = torch.from_numpy(im).cuda()

        albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
        albedoBSPreds, roughBSPreds, depthBSPreds = [], [], []
        envmapsPreds, envmapsPredImages, renderedPreds = [], [], []
        cAlbedos = []
        cLights = []

        ################# BRDF Prediction ######################
        inputBatch = imBatches[0]
        x1, x2, x3, x4, x5, x6 = self.encoders[0](inputBatch)

        albedoPred = 0.5 * (self.albedoDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
        normalPred = self.normalDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
        roughPred = self.roughDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
        depthPred = 0.5 * (self.depthDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)

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

        imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                     align_corners=True)
        albedoPredLarge = F.interpolate(albedoPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                        align_corners=True)
        normalPredLarge = F.interpolate(normalPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                        align_corners=True)
        roughPredLarge = F.interpolate(roughPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                       align_corners=True)
        depthPredLarge = F.interpolate(depthPreds[0], [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4], mode='bilinear',
                                       align_corners=True)

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge],
                               dim=1)
        x1, x2, x3, x4, x5, x6 = self.lightEncoders[0](inputBatch)

        # Prediction
        axisPred = self.axisDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
        lambPred = self.lambDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
        weightPred = self.weightDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall)
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred], dim=1)
        envmapsPreds.append(envmapsPred)

        envmapsPredImage, axisPred, lambPred, weightPred = self.output2env.output2env(axisPred, lambPred, weightPred)
        envmapsPredImages.append(envmapsPredImage)

        diffusePred, specularPred = self.renderLayer.forwardEnv(albedoPreds[0], normalPreds[0], roughPreds[0], envmapsPredImages[0])

        diffusePredNew, specularPredNew = cis_network.LSregressDiffSpec(diffusePred, specularPred, imBatchSmall, diffusePred, specularPred)
        renderedPred = diffusePredNew + specularPredNew
        renderedPreds.append(renderedPred)

        cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred)).data.item(), (
                    (torch.sum(specularPredNew)) / (torch.sum(specularPred))).data.item()
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

        inputBatch = torch.cat(
            [imBatches[1], albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge, diffusePredLarge,
             specularPredLarge], dim=1)

        x1, x2, x3, x4, x5, x6 = self.encoders[1](inputBatch)
        albedoPred = 0.5 * (self.albedoDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)
        normalPred = self.normalDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
        roughPred = self.roughDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
        depthPred = 0.5 * (self.depthDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)

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

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge, 0.5 * (normalPredLarge + 1), 0.5 * (roughPredLarge + 1), depthPredLarge],
                               dim=1)
        x1, x2, x3, x4, x5, x6 = self.lightEncoders[1](inputBatch, envmapsPred)

        # Prediction
        axisPred = self.axisDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
        lambPred = self.lambDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
        weightPred = self.weightDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred], dim=1)
        envmapsPreds.append(envmapsPred)

        envmapsPredImage, axisPred, lambPred, weightPred = self.output2env.output2env(axisPred, lambPred, weightPred)
        envmapsPredImages.append(envmapsPredImage)

        diffusePred, specularPred = self.renderLayer.forwardEnv(albedoPreds[1], normalPreds[1], roughPreds[1], envmapsPredImages[1])

        diffusePredNew, specularPredNew = cis_network.LSregressDiffSpec(diffusePred, specularPred, imBatchSmall, diffusePred, specularPred)

        renderedPre = diffusePredNew + specularPredNew
        renderedPreds.append(renderedPred)

        cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred)).data.item(), (
                    (torch.sum(specularPredNew)) / (torch.sum(specularPred))).data.item()
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
            albedoBSPred, albedoConf = self.albedoBSs[n](imBatches[n], albedoPreds[n].detach(), albedoPreds[n])
            albedoBSPreds.append(albedoBSPred)
            roughBSPred, roughConf = self.roughBSs[n](imBatches[n], albedoPreds[n].detach(), 0.5 * (roughPreds[n] + 1))
            roughBSPred = torch.clamp(2 * roughBSPred - 1, -1, 1)
            roughBSPreds.append(roughBSPred)
            depthBSPred, depthConf = self.depthBSs[n](imBatches[n], albedoPreds[n].detach(), depthPreds[n])
            depthBSPreds.append(depthBSPred)

        dict_forward = {}
        dict_forward['albedo'] = albedoBSPreds[-1]
        dict_forward['normal'] = normalPreds[-1]
        dict_forward['rough'] = roughBSPreds[-1]
        dict_forward['depth'] = depthBSPreds[-1]
        return dict_forward
        # # Save the envmapImages
        # for n in range(0, len(envmapsPredImages)):
        #     envmapsPredImage = envmapsPredImages[n].data.cpu().numpy().squeeze()
        #     envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])
        #
        #     # Flip to be conincide with our dataset
        #     # np.savez_compressed(envmapPredImNames[n], env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
        #     np.savez(envmapPredImNames[n], env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
        #     utils.writeEnvToFile(envmapsPredImages[n], 0, envmapPredImNames[n], nrows=24, ncols=16)
        #
        #     for n in range(0, len(envmapsPreds)):
        #         envmapsPred = envmapsPreds[n].data.cpu().numpy()
        #         np.save(envmapsPredSGNames[n], envmapsPred)
        #         shading = utils.predToShading(envmapsPred, SGNum=12)
        #         shading = shading.transpose([1, 2, 0])
        #         shading = shading / np.mean(shading) / 3.0
        #         shading = np.clip(shading, 0, 1)
        #         shading = (255 * shading ** (1.0 / 2.2)).astype(np.uint8)
        #         cv2.imwrite(shadingNames[n], shading[:, :, ::-1])
        #
        #     for n in range(0, len(cLights)):
        #         io.savemat(cLightNames[n], {'cLight': cLights[n]})
        #
        #     # Save the rendered image
        #     for n in range(0, len(renderedPreds)):
        #         renderedPred = renderedPreds[n].data.cpu().numpy().squeeze()
        #         renderedPred = renderedPred.transpose([1, 2, 0])
        #         renderedPred = (renderedPred / renderedPred.max()) ** (1.0 / 2.2)
        #         renderedPred = cv2.resize(renderedPred, (nw, nh), interpolation=cv2.INTER_LINEAR)
        #         # np.save(renderedNames[n], renderedPred )
        #
        #         renderedPred = (np.clip(renderedPred, 0, 1) * 255).astype(np.uint8)
        #         cv2.imwrite(renderedImNames[n], renderedPred[:, :, ::-1])



