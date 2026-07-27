import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm
import numpy as np

def compute_ref_warping_error(orig_frames, gen_frames, face_masks, device="cuda", fb_thresh=1.0, weights_path=None, video_length=None, random_seed=None):
    """
    Compute reference-style warping error with face mask constraint.

    Args:
        orig_frames: list[torch.Tensor], each (H,W,3), float32 in [0,1]
        gen_frames:  list[torch.Tensor], same shape/len as orig_frames
        face_masks:  list[torch.Tensor], each (H,W,3), binary {0,1}
        device:      str, e.g. "cuda" or "cpu"
        fb_thresh:   float, forward-backward consistency threshold (pixels)
        weights_path:str or None, path to local RAFT weights (.pth). If None, use torchvision's default.
        video_length:int or None, if specified, randomly crop video to this length
        random_seed: int or None, random seed for reproducible cropping

    Returns:
        dict with short_list, long_list, mean_short, mean_long, mean_total
    """

    T = len(orig_frames)
    
    # Apply random cropping if video_length is specified
    if video_length is not None and video_length < T:
        if random_seed is not None:
            np.random.seed(random_seed)
        
        start_idx = np.random.randint(0, T - video_length + 1)
        end_idx = start_idx + video_length
        
        orig_frames = orig_frames[start_idx:end_idx]
        gen_frames = gen_frames[start_idx:end_idx]
        face_masks = face_masks[start_idx:end_idx]
        T = video_length

    H, W, _ = orig_frames[0].shape

    # stack tensors
    orig_frames = torch.stack([f.permute(2,0,1) for f in orig_frames]).unsqueeze(0).to(device)
    gen_frames  = torch.stack([f.permute(2,0,1) for f in gen_frames]).to(device)
    face_masks  = torch.stack([m.permute(2,0,1) for m in face_masks]).to(device)

    # -------- load RAFT large --------
    if weights_path is None:
        weights = Raft_Large_Weights.C_T_SKHT_V2
        model = raft_large(weights=weights, progress=True).to(device).eval()
    else:
        model = raft_large(weights=None).to(device).eval()
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    # -------- helper functions --------
    def warp(x, flow):
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device),
            torch.arange(W, device=flow.device),
            indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).repeat(B,1,1,1)
        flow_ = flow.permute(0,2,3,1)
        coords = grid + flow_
        coords_x = 2.0 * coords[...,0] / max(W-1,1) - 1.0
        coords_y = 2.0 * coords[...,1] / max(H-1,1) - 1.0
        grid_norm = torch.stack((coords_x, coords_y), -1)
        return F.grid_sample(x, grid_norm, align_corners=True, padding_mode='border')

    def occ_mask(flow_fw, flow_bw):
        B, _, H, W = flow_fw.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow_fw.device),
            torch.arange(W, device=flow_fw.device),
            indexing="ij"
        )
        coords = torch.stack((grid_x, grid_y), 0).float().unsqueeze(0)
        coords_fw = coords + flow_fw
        coords_bw = warp(coords_fw, flow_bw)
        diff = (coords_bw - coords).norm(dim=1, keepdim=True)
        mask = (diff < fb_thresh).float()
        return mask

    # -------- compute errors --------
    short_list, long_list = [], []
    with torch.inference_mode():
        for t in tqdm(range(1, T), desc="Computing ref warping error"):
            for s, container in [(t-1, short_list), (0, long_list)]:
                img1 = orig_frames[:,s]
                img2 = orig_frames[:,t]
                flow_fw = model(img1, img2)[-1]
                flow_bw = model(img2, img1)[-1]


                # masks
                M_occ = occ_mask(flow_fw, flow_bw)
                face_s = face_masks[s].unsqueeze(0)
                face_t = face_masks[t].unsqueeze(0)
                face_s_warp = warp(face_s, flow_fw)
                face_inter = (face_s_warp>0.5) * (face_t>0.5)
                M_final = (M_occ * face_inter).squeeze(0)

                # warp gen_s to t
                gen_s = gen_frames[s].unsqueeze(0)
                gen_t = gen_frames[t].unsqueeze(0)
                gen_s_warp = warp(gen_s, flow_fw)

                diff = (gen_t - gen_s_warp).abs().mean(1,keepdim=True)
                masked = diff * M_final
                if M_final.sum() > 0:
                    err = masked.sum() / (M_final.sum()+1e-8)
                    container.append(err.item())
                else:
                    container.append(float("nan"))

    mean_short = torch.tensor([x for x in short_list if not torch.isnan(torch.tensor(x))]).mean().item()
    mean_long  = torch.tensor([x for x in long_list  if not torch.isnan(torch.tensor(x))]).mean().item()
    mean_total = (mean_short + mean_long)/2

    return {
        "short_list": short_list,
        "long_list": long_list,
        "mean_short": mean_short,
        "mean_long": mean_long,
        "mean_total": mean_total
    }
