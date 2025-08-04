import torch
import torch.nn as nn
from .utils import *
from .torch_geometry import get_perspective_transform

class DenseMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2
        self.W = 5
        self.r = 8/2//2.

    # @torch.no_grad()
    def get_fine_match_dense(self, data, dense_num = None, four_point_disp = None, update_fmatch = True, get_norm = False, mask = None):
        # M = four_point_disp.shape[0]
        # M = data['mkpts0_c'].shape[0]
        four_point_disp = four_point_disp if four_point_disp is not None else data['flow_predictions'][-1]
        # four_point_disp = four_point_disp[:M]
        M = four_point_disp.shape[0]

        if M == 0:
            data.update({
                "mkpts0_f_dense": four_point_disp.new_empty((0, 2)),
                "mkpts1_f_dense": four_point_disp.new_empty((0, 2))})
            return

        self.mask = torch.ones(M) > 0 if mask is None else mask
        M = self.mask.sum()
        H = self.get_homo_now_k(four_point_disp[self.mask], k=1)
        center_pix = (self.W-1)/2.0
        points = torch.cat((torch.ones((1,1))*center_pix, torch.ones((1,1))*center_pix, torch.ones((1,1))),
                dim=0).unsqueeze(0).repeat(M, 1, 1).to(four_point_disp.device)# [M,2,1] only one point
        
        r = self.r # self.W
        dense_num = dense_num if dense_num else 2*r+1
        if dense_num == 1:
            dense_points = points[:,:2,:]
        else:
            dense_points = self._coords_dense(points[:,:2,0],r=r,dense_num=dense_num).view(M, -1, 2).permute(0,2,1)
        warp_dense_points = H.bmm(torch.cat((dense_points,torch.ones((M,1,dense_points.shape[-1])).to(four_point_disp.device)),dim=1))
        warp_dense_points = warp_dense_points / warp_dense_points[:, 2, :].unsqueeze(1)

        if get_norm:
            warp_dense_points = warp_dense_points[:,:2,0]
            warp_dense_points_norm = (warp_dense_points-center_pix)/(center_pix)
            return warp_dense_points_norm

        dense_points = dense_points.permute(0,2,1).reshape(M,dense_num,dense_num,2)
        warp_dense_points = warp_dense_points[:,:2,:].permute(0,2,1).reshape(M,dense_num,dense_num,2)
        self._get_fine_match_dense(dense_points, warp_dense_points, data, update_fmatch)


    # @torch.no_grad()
    def get_homo_now_k(self, four_point, k = 1):
        # N,_,_,_ = four_point.shape
        N = four_point.shape[0]
        # h, w = h//k, w//k
        h, w = self.W, self.W
        four_point = four_point / k # four_point is at original size， coordinate is at feature map size
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
        four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = get_perspective_transform(four_point_org, four_point_new)

        return H
    
    # @torch.no_grad()
    def _coords_dense(self, grid_pt_i, r = 2, dense_num = None):
        """
        Args:
            grid_pt_i: torch.Tensor, [N, 2]
        """
        dense_num = dense_num if dense_num else 2*r+1
        dx = torch.linspace(-r, r, dense_num)  # [-r, r]
        dy = torch.linspace(-r, r, dense_num)
        delta = torch.stack(torch.meshgrid(dy, dx,indexing='xy'), axis=-1).to(grid_pt_i.device) # h*w*2 (x,y)
        centroid_lvl = grid_pt_i.reshape(-1, 1, 1, 2) # N * h0, w0
        delta_lvl = delta.view(1, dense_num, dense_num , 2) 
        coords_lvl = centroid_lvl + delta_lvl # [M_, 2, 2, 2]
        return coords_lvl

    # @torch.no_grad()
    def _get_fine_match_dense(self, dense_points, warp_dense_points, data, update_fmatch = True):        
        W, WW, scale = self.W, self.W**2, self.scale

        # mkpts0_f and mkpts1_f
        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        if isinstance(scale0, int):
            mkpts0_f = data['mkpts0_c'].reshape(-1, 1, 1, 2)[self.mask] + ((dense_points-W // 2) * scale0)
        else:
            scale0 = scale0[self.mask] # [:len(data['mconf'])]
            mkpts0_f = data['mkpts0_c'].reshape(-1, 1, 1, 2)[self.mask] + ((dense_points-W // 2) * scale0.view(-1,1,1,2))
        mkpts0_f = mkpts0_f.view(-1,2)
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        if isinstance(scale1, int):
            mkpts1_f = data['mkpts1_c'].reshape(-1, 1, 1, 2)[self.mask] + ((warp_dense_points-W // 2) * scale1)
        else:
            scale1 = scale1[self.mask] # [:len(data['mconf'])]
            mkpts1_f = data['mkpts1_c'].reshape(-1, 1, 1, 2)[self.mask] + ((warp_dense_points-W // 2) * scale1.view(-1,1,1,2))
        mkpts1_f = mkpts1_f.reshape(-1,2)

        data.update({
            "mkpts0_f_dense": mkpts0_f,
            "mkpts1_f_dense": mkpts1_f
        })

        if update_fmatch:
            data.update({
                "mkpts0_f": mkpts0_f,
                "mkpts1_f": mkpts1_f
            })


def dense_project(image_source, image_target, mkpts0, mkpts1, device = 'cpu'):
    '''
    Args:
        image_source: np.array, [H, W, C]
        image_target: np.array, [H, W, C]
        mkpts0: np.array, [N, 2]
        mkpts1: np.array, [N, 2]
    Returns:
        figures (Dict[str, List[plt.figure]]
    '''
    Ho, Wo = image_target.shape[:2]
    out = torch.ones((Ho, Wo, 2)).to(device)*-1
    out[mkpts1[:, 1].clip(0, Ho-1), mkpts1[:, 0].clip(0, Wo-1)] = torch.tensor(mkpts0)
    xgrid, ygrid = out.split([1,1], dim=-1)

    Hi, Wi = image_source.shape[:2]
    xgrid = 2*xgrid/(Wi-1) - 1
    ygrid = 2*ygrid/(Hi-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).unsqueeze(0) 
    def get_img_tensor(img):
        if len(img.shape) == 2:
            return torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)/255.
        elif len(img.shape) == 3:
            return torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)/255.
        get_img_tensor(image_source)
    img_warped = F.grid_sample(get_img_tensor(image_source), grid, align_corners=True) 

    out = torch.ones((Ho, Wo, 2)).to(device)*-1
    out[mkpts1[:, 1].clip(0, Ho-1), mkpts1[:, 0].clip(0, Wo-1)] = torch.tensor(mkpts1)
    xgrid, ygrid = out.split([1,1], dim=-1)
    xgrid = 2*xgrid/(Wo-1) - 1
    ygrid = 2*ygrid/(Ho-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).unsqueeze(0)  # [1024, 9, 9, 2]
    img_src_sampled = F.grid_sample(get_img_tensor(image_target), grid, align_corners=True) 

    return img_warped, img_src_sampled

def dense_project_tensor(image_source, image_target, mkpts0, mkpts1, batch = None):
    '''
    Args:
        image_source: np.array, [B, C, H, W]
        image_target: np.array, [B, C, H, W]
        mkpts0: np.array, [N, 2]
        mkpts1: np.array, [N, 2]
    Returns:
        figures (Dict[str, List[plt.figure]]
    '''
    if batch is not None:
        image_source = batch['image0']
        image_target = batch['image1']
        mkpts0 = batch['mkpts0_f_dense']
        mkpts1 = batch['mkpts1_f_dense']
    device = image_source.device
    Ho, Wo = image_target.shape[-2:]
    out = torch.ones((Ho, Wo, 2)).to(device)*-1
    out[mkpts1[:, 1].clip(0, Ho-1).long(), mkpts1[:, 0].clip(0, Wo-1).long()] = mkpts0 # .long().float()
    xgrid, ygrid = out.split([1,1], dim=-1)

    Hi, Wi = image_source.shape[-2:]
    xgrid = 2*xgrid/(Wi-1) - 1
    ygrid = 2*ygrid/(Hi-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).unsqueeze(0) 

    img_warped = F.grid_sample(image_source, grid, align_corners=True) 

    out = torch.ones((Ho, Wo, 2)).to(device)*-1
    out[mkpts1[:, 1].clip(0, Ho-1).long(), mkpts1[:, 0].clip(0, Wo-1).long()] = mkpts1 # .long().float()
    xgrid, ygrid = out.split([1,1], dim=-1)
    xgrid = 2*xgrid/(Wo-1) - 1
    ygrid = 2*ygrid/(Ho-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).unsqueeze(0)  # [1024, 9, 9, 2]
    img_src_sampled = F.grid_sample(image_target, grid, align_corners=True) 

    return img_warped, img_src_sampled