import torch
import torch.nn as nn

from .corr import CorrBlock
from .update import GMA

# from loguru import logger
from .utils.torch_geometry import get_perspective_transform
from .utils.utils import *

autocast = torch.cuda.amp.autocast


class FineHomo(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()

        self.padding = True
        self.radius = 1
        self.scale = 2
        self.homo_dense = False
        self.W = self.window_size = window_size
        self.cat_std = False

        in_dim = (2 * self.radius + 1) ** 2 * 2 + 2  #
        self.update_block_4 = GMA(self.W, in_dim)  # 5 3*3
        self.pad_num = 3200
        self.thresh = np.array([600, 1200, 3200, 4800, 6400, 8000])  # , 9600])
        self.coords0, self.coords1 = {}, {}
        self.points = {}
        self.four_point_disp = {}
        self.get_nan = False

        if self.padding:
            self.initialize_coords()
        # For Dense Match
        # self.config = config
        # if self.homo_dense:
        #     self.Dense_Matcher = DenseMatch(r=1, config=config)

    def initialize_coords(self):
        for M in self.thresh:
            if M not in self.coords0:
                N, H, W = M, self.W, self.W
                coords0 = coords_grid(N, H, W)  # [batch,2, H, W]
                coords1 = coords0.clone()
                self.coords0[M], self.coords1[M] = coords0, coords1
                # logger.info('\n'+str(M)+'+'*10)
                self.four_point_disp[M] = torch.zeros((M, 2, 2, 2))

    def get_flow_now_k(self, four_point, k=1):
        H = self.get_homo_now_k(four_point, k=k)
        N = four_point.shape[0]
        h, w = self.W, self.W
        # gridy, gridx = torch.meshgrid(torch.linspace(0, h-1, steps=h), torch.linspace(0,w-1, steps=w),indexing='ij')
        # points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
        #                    dim=0).unsqueeze(0).repeat(N, 1, 1).to(four_point.device)
        if N not in self.points:
            self.points[N] = torch.cat(
                (
                    self.coords0[N].view(N, 2, -1),
                    torch.ones((N, w * h)).unsqueeze(1).to(four_point.device),
                ),
                dim=1,
            )
        points = self.points[N].clone()
        points_new = H.bmm(points)  # (N,3,3) (N,3,w*h)
        safe_denominator = points_new[:, 2, :].unsqueeze(1) + 1e-8
        # if ((abs(safe_denominator) < 1e-7).any() or torch.isnan(safe_denominator).any()) and self.get_nan == False:
        #     logger.info(f'safe_denominator: {safe_denominator.min()}, {abs(safe_denominator).min()}')
        points_new = points_new / safe_denominator
        points_new = points_new[:, 0:2, :]
        flow = torch.cat(
            (
                points_new[:, 0, :].reshape(N, h, w).unsqueeze(1),
                points_new[:, 1, :].reshape(N, h, w).unsqueeze(1),
            ),
            dim=1,
        )

        return flow, H

    def get_homo_now_k(self, four_point, k=1):
        # N,_,_,_ = four_point.shape
        N = four_point.shape[0]
        # h, w = h//k, w//k
        h, w = self.W, self.W
        four_point = (
            four_point / k
        )  # four_point is at original size， coordinate is at feature map size
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h - 1])
        four_point_org[:, 1, 1] = torch.Tensor([w - 1, h - 1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = get_perspective_transform(four_point_org, four_point_new)
        # if (torch.isnan(H).any() or torch.isinf(H).any()) and self.get_nan == False:
        #     logger.info(f'Homo_nan: {torch.isnan(H).sum()}, Homo_inf: {torch.isinf(H).sum()}')
        #     center =four_point_new.mean(dim=1)
        #     tmp = (four_point_new - center.unsqueeze(dim=1)).reshape(-1,2)
        #     angles = torch.arctan2(tmp[:, 1], tmp[:, 0]).view(-1,4)
        #     mask = (torch.argsort(angles,dim=-1) == torch.tensor([0, 1, 3, 2]).to(angles.device)).all(dim=-1)
        #     logger.info(f'mask_sum: {mask.sum()}, mask_len: {mask.shape}')

        return H

    def initialize_flow_k(self, img, k=4):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // k, W // k).to(
            img.device
        )  # [batch,2, H, W]
        coords1 = coords_grid(N, H // k, W // k).to(img.device)

        return coords0, coords1  # [x,y]

    def forward(self, feat_f0_unfold, feat_f1_unfold, num_iters=1, init=None):
        """
        Input:
            feat_f0_unfold: (torch.Tensor): (M, W*W, dim) dim=128
        """
        # 0. Normalize input data
        # self.sz = data['hw0_i'] # torch.Size([480, 640])
        # self.W = data['W']
        # pad_num = self.pad_num
        M_, dim, _, _ = feat_f0_unfold.shape

        # if M_ == 0:
        #     return {"fine_reg_biases": feat_f0_unfold.new_empty((0, 2))}

        if self.padding:
            # bz = data['image0'].shape[0]
            # pad_size = 0
            try:
                # pad_num = self.thresh[np.where((self.thresh/ (M_/bz) >= 1.0))[0][0]]
                pad_num = self.thresh[np.where(self.thresh / (M_) >= 1.0)[0][0]]
            except:
                pad_num = self.thresh[-1]
            # if M_ > bz * 3200: logger.info(f'bz too big in fine_homo {M_}/{bz}')
            # if M_ < bz * pad_num:
            if pad_num > M_:
                # 如果当前批次大小小于 M，则进行 padding
                # pad_size = bz * pad_num - M_
                pad_size = pad_num - M_
                padding = torch.zeros(
                    pad_size,
                    dim,
                    self.W,
                    self.W,
                    device=feat_f0_unfold.device,
                    dtype=feat_f0_unfold.dtype,
                )
                feat_f0_unfold = torch.cat([feat_f0_unfold, padding], dim=0)
                feat_f1_unfold = torch.cat([feat_f1_unfold, padding], dim=0)
        M = feat_f0_unfold.shape[0]

        # import pdb;pdb.set_trace()

        # corner case: if no coarse matches found
        # if M_ == 0:
        #     assert self.training == False, "M is always >0, when training, see coarse_matching.py"
        #     data.update({
        #         'expec_f': torch.empty(0, 3, device=feat_f0_unfold.device),
        #         'mkpts0_f': data['mkpts0_c'],
        #         'mkpts1_f': data['mkpts1_c'],
        #         'flow_predictions': torch.empty(0, 2, 2, 2, device=feat_f0_unfold.device),
        #     })
        #     return

        # 1. Using Backbone to Obtain Feature Maps
        sim_matrix = torch.einsum(
            "mcl,mcr->mlr",
            feat_f0_unfold.flatten(start_dim=-2),
            feat_f1_unfold.flatten(start_dim=-2),
        )
        corr = sim_matrix.view(-1, self.W, self.W, 1, self.W, self.W)

        fmap1, fmap2 = feat_f0_unfold, feat_f1_unfold
        # fmap1 = feat_f0_unfold.view(M, self.W, self.W, dim).permute(
        #     0, 3, 1, 2
        # )  # [M, W,W,dim]]
        # fmap2 = feat_f1_unfold.view(M, self.W, self.W, dim).permute(
        #     0, 3, 1, 2
        # )  # # [320, 16, 16]]
        # sz = data['hw0_f'] # torch.Size([240, 320])

        # 2. Calculate Correlation Matrix
        corr_fn = CorrBlock(
            fmap1, fmap2, num_levels=2, radius=self.radius, corr=corr
        )  # radius = 8

        if self.padding:
            if M not in self.coords0:
                # self.coords0, self.coords1 = self.initialize_flow_k(fmap1, k=1)
                coords0, coords1 = self.initialize_flow_k(fmap1, k=1)
                self.coords0[M], self.coords1[M] = coords0, coords1
                # logger.info('\n'+str(M)+'+'*10)
                self.four_point_disp[M] = torch.zeros((M, 2, 2, 2)).to(
                    fmap1.device
                )
            coords0, coords1 = (
                self.coords0[M].to(fmap1),
                self.coords1[M].to(fmap1),
            )
            four_point_disp = self.four_point_disp[M].to(fmap1)
        else:
            coords0, coords1 = self.initialize_flow_k(fmap1, k=1)
            four_point_disp = torch.zeros((M, 2, 2, 2)).to(fmap1.device)

        # 3. Recurrent Homography Estimation
        flow_predictions = []  ## for train
        for itr in range(num_iters):
            corr = corr_fn(coords1)  # batch,channel,H,W  correlation
            flow = coords1 - coords0  # [batch,2, H, W] mean x, y
            # with autocast(enabled=self.args.mixed_precision):

            # with torch.backends.cudnn.flags(enabled=False):
            delta_four_point = self.update_block_4(
                corr, flow
            )  # input shape: [b,2+channel,H,W] , output shape [b,2,2,2]

            four_point_disp = four_point_disp + delta_four_point
            if num_iters > itr + 1:
                coords1, H = self.get_flow_now_k(
                    four_point_disp, k=1
                )  # self.sz[-1]//sz[-1])
            else:
                H = self.get_homo_now_k(four_point_disp, k=1)
            H = H.to(corr.dtype)
            flow_predictions.append(four_point_disp[:M_])  ## for train

        # del_w = four_point_disp[:M_][:,0,0,1] - four_point_disp[:M_][:,0,0,0] + self.W - 1
        # if del_w.mean()/self.W < 0.0 and(not scaled):
        #     # logger.info('='*30)
        #     # b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
        #     b_ids, i_ids, j_ids = data["idxes"]
        #     (h0, w0), (h1, w1) = size0, size1
        #     delta = torch.tensor([
        #         -w0-1,-w0,-w0+1,
        #         -1,torch.nan,1,
        #         w0-1,w0,w0+1
        #     ]).to(i_ids.device).reshape(1,3,3,1)
        #     i_ids_lvl = i_ids.reshape(-1, 1, 1, 1) # N * h0, w0
        #     i_ids = (i_ids_lvl + delta).reshape(-1) # [M_, 2, 2, 2]
        #     i_ids = i_ids[~torch.isnan(i_ids)]
        #     mask = ~torch.isin(i_ids, data["idxes"][1])
        #     mask[i_ids < 1] = False
        #     mask[i_ids > (h0 * w0 - 2)] = False
        #     data['scores'] = torch.concat((data['scores'], data['scores'].repeat_interleave(repeats=8)[mask]), dim = -1)
        #     b_ids = torch.concat((b_ids, b_ids.repeat_interleave(repeats=8)[mask]), dim = -1).long()
        #     i_ids = torch.concat((data["idxes"][1], i_ids[mask]), dim = -1).long()
        #     j_ids = torch.concat((j_ids, j_ids.repeat_interleave(repeats=8)[mask]), dim = -1).long()
        #     # logger.info(f"{data['i_ids'].shape}, {i_ids.shape}")
        #     data["idxes"] = b_ids, i_ids, j_ids
        #     # scale = data['hw0_i'][0] / data['hw0_c'][0]
        #     # scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        #     # scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        #     mkpts0_c = torch.stack(
        #         [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
        #         dim=1)
        #     mkpts1_c = torch.stack(
        #         [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
        #         dim=1)
        #     data['points0'] = mkpts0_c
        #     data['points1'] = mkpts1_c
        #     # logger.info('-'*30)
        #     # logger.info( '#'.join([item[0] for item in data['pair_names']]))
        #     return False

        # 4. Output
        # coords1,H = self.get_flow_now_k(four_point_disp, k=1)
        if init is not None:
            points, r_points = ((self.W - 1) * init).chunk(2, dim=1)
            points = torch.cat(
                [points, torch.ones_like(points[:, [0]])], dim=1
            )[:, :, None]
            if self.padding:
                _points = points.new_zeros((M, 3, 1))
                _points[:M_] = points
                points = _points
        else:
            center_pix = (self.W - 1) / 2.0
            points = (
                torch.cat(
                    (
                        torch.ones((1, 1)) * center_pix,
                        torch.ones((1, 1)) * center_pix,
                        torch.ones((1, 1)),
                    ),
                    dim=0,
                )
                .unsqueeze(0)
                .repeat(M, 1, 1)
                .to(four_point_disp.device)
            )  # [M,2,1] only one point
            r_points = (self.W - 1) / 2.0
        warp_points = H.bmm(points)  # [3840, 2]
        safe_denominator = warp_points[:, 2, :].unsqueeze(1) + 1e-8
        # safe_denominator = denominator.clone().clamp(min=1e-8)
        warp_points = warp_points / safe_denominator
        warp_points = warp_points[:, :2, 0]
        offset = (warp_points[:M_] - r_points) / (self.W // 2)
        # offset = offset[:M_]
        # if self.cat_std:
        #     self.get_coor_std(corr_fn, M, warp_points,M_, data,offset)
        # else:
        # data.update({'expec_f': offset}) # offset torch.cat([offset, sim_matrix_exp.view(M,-1)], -1)

        # H_inverse = H.inverse()
        # warp_points_inverse = H_inverse.bmm(points) # [3840, 2]
        # safe_denominator = warp_points_inverse[:, 2, :].unsqueeze(1) + 1e-8
        # # safe_denominator = denominator.clone().clamp(min=1e-8)
        # warp_points_inverse = warp_points_inverse / safe_denominator
        # warp_points_inverse = warp_points_inverse[:,:2,0]
        # offset_inverse = (warp_points_inverse-self.W//2)/(self.W//2)
        # offset_inverse = offset_inverse[:M_]
        # data.update({'expec_f_inverse': offset_inverse})

        # data.update({'flow_predictions': flow_predictions})
        # if (torch.isnan(offset).any() or torch.isinf(offset).any()) and self.get_nan == False:
        #     feat_f0_unfold, feat_f1_unfold, data
        #     nan_data = {
        #         'feat_f0_unfold': feat_f0_unfold,
        #         'feat_f1_unfold': feat_f1_unfold,
        #         'sim_matrix': sim_matrix,
        #         'data': data}
        #     self.get_nan = True
        #     torch.save(nan_data, 'nan_data.pt')
        #     logger.info(f'safe_denominator: {safe_denominator.min()}, {abs(safe_denominator).min()}')
        #     logger.info(f'flow_nan: {torch.isnan(four_point_disp).sum()}, flow_inf: {torch.isinf(four_point_disp).sum()}')
        #     logger.info(f'M_num: {M}, correlation_nan: {torch.isnan(sim_matrix).sum()}, correlation_inf: {torch.isinf(sim_matrix).sum()}')

        # if M_ < 0:
        #     self.Dense_Matcher = DenseMatch(r=1, config=self.config)
        #     dense_num = 2*4+1 # =1 mean center point
        #     mask = data['mconf'] > 0.0
        #     self.Dense_Matcher.get_fine_match_dense(data, dense_num=dense_num, update_fmatch = True, mask=mask) # [M*dense_num^2, 2]
        #     mask_tmp = data['mconf'][mask].repeat_interleave(repeats=dense_num**2)
        #     data['mconf'] = mask_tmp
        # else:
        #     if self.homo_dense:
        #         self.Dense_Matcher = DenseMatch(r=1.6, config=self.config)
        #         dense_num = 2*1+1 # =1 mean center point
        #         mask = data['mconf'] > 0.0
        #         self.Dense_Matcher.get_fine_match_dense(data, dense_num=dense_num, update_fmatch = True, mask=mask) # [M*dense_num^2, 2]
        #         mask_tmp = data['mconf'][mask].repeat_interleave(repeats=dense_num**2)
        #         data['mconf'] = mask_tmp
        #         # self.dense_num = 4
        #         # dense_points = self.coords_dense(points[:,:2,0],r=3).view(M, self.dense_num**2, 2).permute(0,2,1)
        #         # warp_dense_points = H.bmm(torch.cat((dense_points,torch.ones((M,1,self.dense_num**2)).to(four_point_disp.device)),dim=1))
        #         # dense_points = dense_points[:M_].permute(0,2,1).reshape(M_,self.dense_num,self.dense_num,2)
        #         # warp_dense_points = warp_dense_points / warp_dense_points[:, 2, :].unsqueeze(1)
        #         # warp_dense_points = warp_dense_points[:M_,:2,:].permute(0,2,1).reshape(M_,self.dense_num,self.dense_num,2)
        #         # self.get_fine_match_dense(dense_points, warp_dense_points, data)
        #     else:
        # self.get_fine_match(offset, data)
        return {"fine_reg_biases": offset, "flow_predictions": flow_predictions}

    # @torch.no_grad()
    def coords_dense(self, grid_pt_i, r=2):
        """
        Args:
            grid_pt_i: torch.Tensor, [N, 2]
        """
        dx = torch.linspace(-1, r - 1, self.dense_num)  # [-r, r]
        dy = torch.linspace(-1, r - 1, self.dense_num)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="xy"), axis=-1).to(
            grid_pt_i.device
        )  # h*w*2 (x,y)
        centroid_lvl = grid_pt_i.reshape(-1, 1, 1, 2)  # N * h0, w0
        delta_lvl = delta.view(1, self.dense_num, self.dense_num, 2)
        coords_lvl = centroid_lvl + delta_lvl  # [M_, 2, 2, 2]
        return coords_lvl

    # @torch.no_grad()
    def get_fine_match_dense(self, dense_points, warp_dense_points, data):
        W, WW, scale = self.W, self.W**2, self.scale

        # mkpts0_f and mkpts1_f
        # scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale

        b_ids, i_ids, j_ids = data["b_ids"], data["i_ids"], data["j_ids"]
        mkpts0_c = (
            torch.stack(
                [i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]], dim=1
            )
            * 4
        )
        mkpts1_c = (
            torch.stack(
                [j_ids % data["hw1_c"][1], j_ids // data["hw1_c"][1]], dim=1
            )
            * 4
        )
        mkpts0_f = (
            mkpts0_c.reshape(-1, 1, 1, 2)
            + (dense_points - W // 2)[: len(data["mconf"])]
        )
        mkpts0_f = mkpts0_f.view(-1, 2)
        mkpts1_f = (
            mkpts1_c.reshape(-1, 1, 1, 2)
            + (warp_dense_points - W // 2)[: len(data["mconf"])]
        )
        mkpts1_f = mkpts1_f.reshape(-1, 2)
        h_f, w_f = data["hw0_f"]
        dense_warp = torch.ones(h_f, w_f, 2).to(mkpts0_c.device) * (
            max(h_f, w_f) + 1
        )
        dense_warp[mkpts0_f[:, 1].long(), mkpts0_f[:, 0].long()] = mkpts1_f
        # dense_warp = torch.stack((2 * dense_warp[..., 0] / w_f - 1, 2 * dense_warp[..., 1] / h_f - 1), dim=-1)
        # h_o, w_o = max(data['depth0'].shape[-2:]), max(data['depth0'].shape[-2:])
        # dense_warp = F.interpolate(dense_warp.permute(2,0,1)[None], size=[h_o, w_o],align_corners=False,mode="bilinear",)
        # dense_warp = dense_warp[0].permute(1,2,0)
        # dense_warp = torch.stack((w_o * (dense_warp[..., 0] + 1) / 2, h_o * (dense_warp[..., 1] + 1) / 2), dim=-1)
        # dense_warp[data['mkpts0_c'][:,1].long(),data['mkpts0_c'][:,0].long()]

        # unique_points, inverse_indices, counts = torch.unique(mkpts0_f, return_inverse=True, return_counts=True, dim=0)
        # if isinstance(scale0, int):
        #     mkpts0_f = data['mkpts0_c'].reshape(-1, 1, 1, 2) + ((dense_points-W // 2) * scale0)[:len(data['mconf'])]
        # else:
        #     mkpts0_f = data['mkpts0_c'].reshape(-1, 1, 1, 2) + ((dense_points-W // 2) * scale0.view(-1,1,1,2))[:len(data['mconf'])]
        # mkpts0_f = mkpts0_f.view(-1,2)
        # scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        # if isinstance(scale1, int):
        #     mkpts1_f = data['mkpts1_c'].reshape(-1, 1, 1, 2) + ((warp_dense_points-W // 2) * scale1)[:len(data['mconf'])]
        # else:
        #     mkpts1_f = data['mkpts1_c'].reshape(-1, 1, 1, 2) + ((warp_dense_points-W // 2) * scale1.view(-1,1,1,2))[:len(data['mconf'])]
        # mkpts1_f = mkpts1_f.reshape(-1,2)

        # data.update({
        #     "mkpts0_f": data['mkpts0_c'].long(), #[self.dense_num**2//2:][::self.dense_num**2],
        #     "mkpts1_f": dense_warp[data['mkpts0_c'][:,1].long(),data['mkpts0_c'][:,0].long()]  #[self.dense_num**2//2:][::self.dense_num**2]
        # })
        data.update(
            {
                "mkpts0_f": mkpts0_c
                * scale
                * data["scale0"][
                    data["b_ids"]
                ],  # [self.dense_num**2//2:][::self.dense_num**2],
                "mkpts1_f": mkpts1_f[0::16]
                * scale
                * data["scale1"][
                    data["b_ids"]
                ],  # [self.dense_num**2//2:][::self.dense_num**2]
            }
        )

    # @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, scale = self.W, self.W**2, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data["mkpts0_c"]
        scale1 = (
            scale * data["scale1"][data["b_ids"]] if "scale0" in data else scale
        )
        mkpts1_f = (
            data["mkpts1_c"]
            + (coords_normed * (W // 2) * scale1)[: len(data["mconf"])]
        )

        if self.training and "expec_f_inverse" in data:
            coords_normed_inverse = data["expec_f_inverse"]
            scale0 = (
                scale * data["scale0"][data["b_ids"]]
                if "scale0" in data
                else scale
            )
            mkpts0_f_ = (
                data["mkpts0_c"]
                + (coords_normed_inverse * (W // 2) * scale0)[
                    : len(data["mconf"])
                ]
            )
            mkpts1_f_ = data["mkpts1_c"]
            mkpts0_f = torch.concat([mkpts0_f, mkpts0_f_], dim=0)
            mkpts1_f = torch.concat([mkpts1_f, mkpts1_f_], dim=0)
            data["mconf"] = torch.concat([data["mconf"], data["mconf"]], dim=0)

        data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})

    def get_coor_std(self, corr_fn, M, warp_points, M_, data, offset):
        corr_map = corr_fn.corr_map.view(M, self.W, self.W, self.W, self.W)[
            :, self.W // 2, self.W // 2, :, :
        ]
        sim_matrix_exp = F.softmax(corr_map.view(M, -1), dim=1)
        positive_indices = 2 * warp_points.clone() / (self.W - 1) - 1
        sim_matrix_exp = F.grid_sample(
            sim_matrix_exp.view(M, 1, self.W, self.W),
            positive_indices.unsqueeze(1).unsqueeze(1),
            align_corners=True,
        )
        data.update({"conf_f": sim_matrix_exp[:M_].view(M_, -1)})

        data.update(
            {
                "expec_f": torch.cat(
                    [offset, sim_matrix_exp[:M_].view(M_, -1)], -1
                )
            }
        )
        # offset torch.cat([offset, sim_matrix_exp.view(M,-1)], -1)
