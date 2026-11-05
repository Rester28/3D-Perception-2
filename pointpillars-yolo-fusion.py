"""
PointPillars (OpenPCDet, KITTI) in puro PyTorch: nessun spconv, nessun operatore CUDA custom.

Uso:
    python pointpillars_inference.py --ckpt pointpillar_7728.pth --bin 000008.bin
    python pointpillars_inference.py --ckpt pointpillar_7728.pth --bin 000008.bin --save out.png --no_show
    python pointpillars_inference.py --ckpt pointpillar_7728.pth --bin 000008.bin --open3d

Dipendenze: torch, numpy, matplotlib (opzionale: open3d per la vista 3D).
Parametri presi da pointpillar.yaml (kitti_models).
"""
import argparse
import math

import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment


EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
Z_NEAR = 0.5  # distanza minima dei corner dalla camera per essere proiettati
# ----------------------------------------------------------------------------
# Configurazione (dal tuo pointpillar.yaml)
# ----------------------------------------------------------------------------
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
PC_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE = [0.16, 0.16, 4.0]
MAX_POINTS_PER_VOXEL = 32
MAX_VOXELS = 40000  # 'test'
NUM_FILTERS_VFE = 64
LAYER_NUMS = [3, 5, 5]
LAYER_STRIDES = [2, 2, 2]
NUM_FILTERS = [64, 128, 256]
UPSAMPLE_STRIDES = [1, 2, 4]
NUM_UPSAMPLE_FILTERS = [128, 128, 128]
DIR_OFFSET = 0.78539
DIR_LIMIT_OFFSET = 0.0
NUM_DIR_BINS = 2
ANCHOR_CFG = [
    dict(size=[3.9, 1.6, 1.56], rotations=[0, 1.57], bottom_height=-1.78),   # Car
    dict(size=[0.8, 0.6, 1.73], rotations=[0, 1.57], bottom_height=-0.6),    # Pedestrian
    dict(size=[1.76, 0.6, 1.73], rotations=[0, 1.57], bottom_height=-0.6),   # Cyclist
]
FEATURE_MAP_STRIDE = 2
SCORE_THRESH = 0.1
NMS_THRESH = 0.01
NMS_PRE_MAXSIZE = 4096
NMS_POST_MAXSIZE = 500

GRID_X = int(round((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0]))  # 432
GRID_Y = int(round((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1]))  # 496
GRID_Z = int(round((PC_RANGE[5] - PC_RANGE[2]) / VOXEL_SIZE[2]))  # 1


# ----------------------------------------------------------------------------
# Voxelizzazione (sostituisce spconv / cumm)
# ----------------------------------------------------------------------------
def voxelize(points, device):
    """points: (N,4) numpy [x,y,z,intensity] -> voxels (V,32,4), num_points (V), coords (V,4) [b,z,y,x]"""
    pts = torch.from_numpy(points).float().to(device)
    pr = torch.tensor(PC_RANGE, device=device)
    vs = torch.tensor(VOXEL_SIZE, device=device)

    mask = ((pts[:, :3] >= pr[:3]) & (pts[:, :3] < pr[3:])).all(dim=1)
    pts = pts[mask]

    c = ((pts[:, :3] - pr[:3]) / vs).floor().long()
    lin = c[:, 2] * (GRID_Y * GRID_X) + c[:, 1] * GRID_X + c[:, 0]

    uniq, inv, counts = torch.unique(lin, return_inverse=True, return_counts=True)
    order = torch.argsort(inv, stable=True)
    inv_sorted = inv[order]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(order.numel(), device=device) - starts[inv_sorted]

    keep = (rank < MAX_POINTS_PER_VOXEL) & (inv_sorted < MAX_VOXELS)
    sel, v, r = order[keep], inv_sorted[keep], rank[keep]

    num_vox = min(uniq.numel(), MAX_VOXELS)
    voxels = torch.zeros(num_vox, MAX_POINTS_PER_VOXEL, 4, device=device)
    voxels[v, r] = pts[sel]
    num_points = counts[:num_vox].clamp(max=MAX_POINTS_PER_VOXEL)

    u = uniq[:num_vox]
    z = u // (GRID_Y * GRID_X)
    y = (u % (GRID_Y * GRID_X)) // GRID_X
    x = u % GRID_X
    coords = torch.stack([torch.zeros_like(x), z, y, x], dim=1)
    return voxels, num_points, coords


# ----------------------------------------------------------------------------
# Modello (stessi nomi dei layer del checkpoint OpenPCDet)
# ----------------------------------------------------------------------------
class PFNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.norm = nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01)

    def forward(self, x):  # x: (V, P, C)
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        return torch.max(x, dim=1, keepdim=True)[0]  # (V,1,C)


class PillarVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFNLayer(10, NUM_FILTERS_VFE)])
        self.vx, self.vy, self.vz = VOXEL_SIZE
        self.x_off = self.vx / 2 + PC_RANGE[0]
        self.y_off = self.vy / 2 + PC_RANGE[1]
        self.z_off = self.vz / 2 + PC_RANGE[2]

    def forward(self, voxels, num_points, coords):
        xyz = voxels[:, :, :3]
        mean = xyz.sum(dim=1, keepdim=True) / num_points.to(voxels.dtype).view(-1, 1, 1)
        f_cluster = xyz - mean

        f_center = torch.zeros_like(xyz)
        f_center[:, :, 0] = xyz[:, :, 0] - (coords[:, 3].to(voxels.dtype).unsqueeze(1) * self.vx + self.x_off)
        f_center[:, :, 1] = xyz[:, :, 1] - (coords[:, 2].to(voxels.dtype).unsqueeze(1) * self.vy + self.y_off)
        f_center[:, :, 2] = xyz[:, :, 2] - (coords[:, 1].to(voxels.dtype).unsqueeze(1) * self.vz + self.z_off)

        feats = torch.cat([voxels, f_cluster, f_center], dim=-1)  # (V,32,10)
        P = feats.shape[1]
        mask = (torch.arange(P, device=feats.device)[None, :] < num_points[:, None]).unsqueeze(-1)
        feats = feats * mask.to(feats.dtype)

        for pfn in self.pfn_layers:
            feats = pfn(feats)
        return feats.squeeze(1)  # (V,64)


def scatter_to_bev(pillar_feats, coords):
    canvas = torch.zeros(pillar_feats.shape[1], GRID_Z * GRID_Y * GRID_X,
                         dtype=pillar_feats.dtype, device=pillar_feats.device)
    idx = coords[:, 1] + coords[:, 2] * GRID_X + coords[:, 3]
    canvas[:, idx.long()] = pillar_feats.t()
    return canvas.view(1, pillar_feats.shape[1] * GRID_Z, GRID_Y, GRID_X)


class BaseBEVBackbone(nn.Module):
    def __init__(self, input_channels=64):
        super().__init__()
        c_in_list = [input_channels, *NUM_FILTERS[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(len(LAYER_NUMS)):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], NUM_FILTERS[idx], 3, stride=LAYER_STRIDES[idx], padding=0, bias=False),
                nn.BatchNorm2d(NUM_FILTERS[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(LAYER_NUMS[idx]):
                layers += [
                    nn.Conv2d(NUM_FILTERS[idx], NUM_FILTERS[idx], 3, padding=1, bias=False),
                    nn.BatchNorm2d(NUM_FILTERS[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ]
            self.blocks.append(nn.Sequential(*layers))
            s = UPSAMPLE_STRIDES[idx]
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(NUM_FILTERS[idx], NUM_UPSAMPLE_FILTERS[idx], s, stride=s, bias=False),
                nn.BatchNorm2d(NUM_UPSAMPLE_FILTERS[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

    def forward(self, x):
        ups = []
        for blk, deblk in zip(self.blocks, self.deblocks):
            x = blk(x)
            ups.append(deblk(x))
        return torch.cat(ups, dim=1)  # (B,384,H/2,W/2)


class AnchorHeadSingle(nn.Module):
    def __init__(self, in_ch=384):
        super().__init__()
        n_anchor = sum(len(a["rotations"]) for a in ANCHOR_CFG)  # 6
        self.conv_cls = nn.Conv2d(in_ch, n_anchor * len(CLASS_NAMES), 1)
        self.conv_box = nn.Conv2d(in_ch, n_anchor * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_ch, n_anchor * NUM_DIR_BINS, 1)

    def forward(self, x):
        cls = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        box = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        d = self.conv_dir_cls(x).permute(0, 2, 3, 1).contiguous()
        return cls, box, d


def generate_anchors(device):
    """Ritorna (H*W*6, 7) nell'ordine usato da OpenPCDet: y, x, classe, rotazione."""
    fx = GRID_X // FEATURE_MAP_STRIDE  # 216
    fy = GRID_Y // FEATURE_MAP_STRIDE  # 248
    per_class = []
    for a in ANCHOR_CFG:
        xs = torch.linspace(PC_RANGE[0], PC_RANGE[3], fx)  # align_center=False
        ys = torch.linspace(PC_RANGE[1], PC_RANGE[4], fy)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (fy, fx)
        rots = torch.tensor(a["rotations"], dtype=torch.float32)
        dx, dy, dz = a["size"]
        anc = torch.zeros(fy, fx, len(rots), 7)
        anc[..., 0] = gx[..., None]
        anc[..., 1] = gy[..., None]
        anc[..., 2] = a["bottom_height"] + dz / 2
        anc[..., 3], anc[..., 4], anc[..., 5] = dx, dy, dz
        anc[..., 6] = rots
        per_class.append(anc)
    anchors = torch.stack(per_class, dim=2)  # (fy, fx, n_class, n_rot, 7)
    return anchors.view(-1, 7).to(device)


class PointPillar(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))
        self.vfe = PillarVFE()
        self.backbone_2d = BaseBEVBackbone(NUM_FILTERS_VFE)
        self.dense_head = AnchorHeadSingle(sum(NUM_UPSAMPLE_FILTERS))
        self.anchors = None

    @torch.no_grad()
    def forward(self, voxels, num_points, coords):
        pf = self.vfe(voxels, num_points, coords)
        bev = scatter_to_bev(pf, coords)
        feat = self.backbone_2d(bev)
        cls, box, d = self.dense_head(feat)

        if self.anchors is None or self.anchors.device != cls.device:
            self.anchors = generate_anchors(cls.device)
        anchors = self.anchors
        n = anchors.shape[0]
        cls = cls.view(n, len(CLASS_NAMES))
        box = box.view(n, 7)
        d = d.view(n, NUM_DIR_BINS)

        # ResidualCoder.decode
        xa, ya, za, dxa, dya, dza, ra = anchors.unbind(-1)
        xt, yt, zt, dxt, dyt, dzt, rt = box.unbind(-1)
        diag = torch.sqrt(dxa ** 2 + dya ** 2)
        boxes = torch.stack([
            xt * diag + xa, yt * diag + ya, zt * dza + za,
            torch.exp(dxt) * dxa, torch.exp(dyt) * dya, torch.exp(dzt) * dza,
            rt + ra,
        ], dim=-1)

        # direction classifier
        period = 2 * math.pi / NUM_DIR_BINS
        dir_labels = torch.argmax(d, dim=-1)
        val = boxes[:, 6] - DIR_OFFSET
        dir_rot = val - torch.floor(val / period + DIR_LIMIT_OFFSET) * period
        boxes[:, 6] = dir_rot + DIR_OFFSET + period * dir_labels.to(boxes.dtype)

        scores, labels = torch.max(torch.sigmoid(cls), dim=-1)
        return boxes, scores, labels + 1  # label 1-based come OpenPCDet


# ----------------------------------------------------------------------------
# NMS ruotato in BEV (sostituisce nms_gpu CUDA) - numpy puro
# ----------------------------------------------------------------------------
def bev_corners(boxes):
    """boxes (N,7+) -> (N,4,2) angoli BEV in ordine antiorario."""
    cx, cy, dx, dy, th = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    local = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])  # (4,2)
    pts = local[None] * np.stack([dx, dy], axis=1)[:, None, :]  # (N,4,2)
    c, s = np.cos(th), np.sin(th)
    x = pts[..., 0] * c[:, None] - pts[..., 1] * s[:, None] + cx[:, None]
    y = pts[..., 0] * s[:, None] + pts[..., 1] * c[:, None] + cy[:, None]
    return np.stack([x, y], axis=-1)


def _poly_area(p):
    if len(p) < 3:
        return 0.0
    x, y = zip(*p)
    return 0.5 * abs(sum(x[i] * y[(i + 1) % len(p)] - x[(i + 1) % len(p)] * y[i] for i in range(len(p))))


def _clip(subject, clipper):
    """Sutherland-Hodgman, poligoni convessi antiorari."""
    out = list(map(tuple, subject))
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def inter(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = a
            x4, y4 = b
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
    return out


def bev_iou(c1, c2):
    inter = _poly_area(_clip(c1, c2))
    if inter <= 0:
        return 0.0
    return inter / (_poly_area(c1) + _poly_area(c2) - inter)


def rotated_nms_bev(boxes, scores, thresh):
    order = np.argsort(-scores)
    corners = bev_corners(boxes)
    centers = boxes[:, :2]
    radius = 0.5 * np.hypot(boxes[:, 3], boxes[:, 4])
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for ii, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        rest = order[ii + 1:]
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        dist = np.linalg.norm(centers[rest] - centers[i], axis=1)
        for j in rest[dist < radius[rest] + radius[i]]:
            if bev_iou(corners[i], corners[j]) > thresh:
                suppressed[j] = True
    return np.array(keep, dtype=np.int64)


def postprocess(boxes, scores, labels, score_thresh=SCORE_THRESH):
    m = scores > score_thresh
    boxes, scores, labels = boxes[m], scores[m], labels[m]
    if scores.numel() == 0:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=np.int64)
    k = min(NMS_PRE_MAXSIZE, scores.numel())
    scores, idx = torch.topk(scores, k)
    boxes, labels = boxes[idx], labels[idx]
    boxes, scores, labels = boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
    keep = rotated_nms_bev(boxes, scores, NMS_THRESH)[:NMS_POST_MAXSIZE]
    return boxes[keep], scores[keep], labels[keep]

# ----------------------------------------------------------------------------
# Caricamento e inferenza
# ----------------------------------------------------------------------------
def load_model(ckpt_path, device):
    model = PointPillar()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def load_kitti_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

def load_kitti_calib(path):
    """Ritorna P2 (3x4) Tr_velo_to_cam (3x4) e R (3x3) come numpy array."""
    with open(path, "r") as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].strip().split()[1:]]).reshape(3, 4)
    Tr = np.array([float(x) for x in lines[5].strip().split()[1:]]).reshape(3, 4)
    R = np.array([float(x) for x in lines[4].strip().split()[1:]]).reshape(3, 3)
    return P2, Tr, R

def detect(model, points, device, score_thresh=SCORE_THRESH):
    voxels, num_points, coords = voxelize(points, device)
    boxes, scores, labels = model(voxels, num_points, coords)
    return postprocess(boxes, scores, labels, score_thresh)


# ----------------------------------------------------------------------------
# Visualizzazione
# ----------------------------------------------------------------------------
COLORS = {1: "lime", 2: "dodgerblue", 3: "orange"}


def visualize_bev(points, boxes, scores, labels, save=None, show=True):
    

    m = ((points[:, 0] >= PC_RANGE[0]) & (points[:, 0] < PC_RANGE[3]) &
         (points[:, 1] >= PC_RANGE[1]) & (points[:, 1] < PC_RANGE[4]))
    p = points[m]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("black")
    # asse orizzontale = -y (sinistra a sinistra), verticale = x (avanti)
    ax.scatter(-p[:, 1], p[:, 0], c=p[:, 2], s=0.5, cmap="gray", vmin=-2.5, vmax=1.0)

    if len(boxes):
        corners = bev_corners(boxes)
        for b, c, s, l in zip(boxes, corners, scores, labels):
            col = COLORS.get(int(l), "red")
            poly = np.vstack([c, c[:1]])
            ax.plot(-poly[:, 1], poly[:, 0], color=col, lw=1.5)
            front = 0.5 * (c[0] + c[3])  # centro del lato frontale (+dx)
            ax.plot([-b[1], -front[1]], [b[0], front[0]], color=col, lw=1.5)
            ax.text(-c[1, 1], c[1, 0], f"{CLASS_NAMES[int(l) - 1]} {s:.2f}", color=col, fontsize=7)

    ax.set_xlim(-PC_RANGE[4], -PC_RANGE[1])
    ax.set_ylim(PC_RANGE[0], PC_RANGE[3])
    ax.set_aspect("equal")
    ax.set_xlabel("-y [m] (sinistra <-> destra)")
    ax.set_ylabel("x [m] (avanti)")
    ax.set_title(f"PointPillars - {len(boxes)} detection")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print("salvato", save)


def boxes_to_corners_3d(boxes):
    """(N,7) -> (N,8,3)"""
    c4 = bev_corners(boxes)  # (N,4,2)
    out = []
    for b, c in zip(boxes, c4):
        zb, zt = b[2] - b[5] / 2, b[2] + b[5] / 2
        bottom = np.hstack([c, np.full((4, 1), zb)])
        top = np.hstack([c, np.full((4, 1), zt)])
        out.append(np.vstack([bottom, top]))
    return np.array(out)

def compute_iou(a, b):  # [x1, y1, x2, y2] top left bottom right
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def visualize_open3d(points, boxes, labels):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms = [pcd]
    
    palette = {1: [0, 1, 0], 2: [0.1, 0.5, 1], 3: [1, 0.6, 0]}
    if len(boxes):
        for corners, l in zip(boxes_to_corners_3d(boxes), labels):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(corners)
            ls.lines = o3d.utility.Vector2iVector(EDGES)
            ls.paint_uniform_color(palette.get(int(l), [1, 0, 0]))
            geoms.append(ls)
    o3d.visualization.draw_geometries(geoms)

""" P2 (3x4) Tr_velo_to_cam (3x4) e R (3x3)"""
def visualize_cam_projection(boxes, labels, P2, Tr, R):
    img = cv2.imread(args.img)
    colors = {1: (0, 255, 0), 2: (255, 128, 25), 3: (0, 153, 255)}  # BGR
    if(len(boxes)):
        corners = boxes_to_corners_3d(boxes)  # (N,8,3)
        n = corners.shape[0]
        pts = np.concatenate([corners.reshape(-1, 3), np.ones((n*8, 1))], axis=1).T  # reshape (N*8,3), il concat fa diventare (N*8,4), il .T fa diventare (4,N*8)
        p2_h = np.vstack([P2, [0, 0, 0, 1]])  # (4,4)
        
        Tr_h = np.vstack([Tr, [0, 0, 0, 1]])  # (4,4)
        
        R_h = np.eye(4)
        R_h[:3, :3] = R #(4,4)
        
        pts = p2_h @ R_h @ Tr_h  @ pts # (4, N*8)
        u =  pts[0, :] / np.clip(pts[2, :], 1e-6, None) 
        v =  pts[1, :] / np.clip(pts[2, :], 1e-6, None) 
        uv = np.stack([u, v], axis=1).reshape(n, 8, 2)  # (N,8,2)
        depth = pts[2, :].reshape(n, 8)  # (N,8)
        for pts, d, l in zip(uv, depth, labels):
            if (d <= 0.1).any():          # corner behind the camera: skip box
                continue
            col = colors.get(int(l), (0, 0, 255))
            for i, j in EDGES:
                p1 = tuple(int(v) for v in pts[i])
                p2 = tuple(int(v) for v in pts[j])
                cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
            # front face (vertices 0,1,4,5): draw a cross to show heading
            cv2.line(img, tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[7]), col, 1, cv2.LINE_AA)
            cv2.line(img, tuple(int(v) for v in pts[3]), tuple(int(v) for v in pts[4]), col, 1, cv2.LINE_AA)       
        return img

def visualize_pointpillar_yolo_fusion(yolo_bb, pp_boxes, pp_scores, pp_labels, P2, Tr, R):
    img = cv2.imread(args.img)
    if(len(pp_boxes) + len(yolo_bb)):
        corners = boxes_to_corners_3d(pp_boxes)  # (N,8,3)
        n = corners.shape[0]
        pts = np.concatenate([corners.reshape(-1, 3), np.ones((n*8, 1))], axis=1).T  # reshape (N*8,3), il concat fa diventare (N*8,4), il .T fa diventare (4,N*8)
        p2_h = np.vstack([P2, [0, 0, 0, 1]])  # (4,4)
        
        Tr_h = np.vstack([Tr, [0, 0, 0, 1]])  # (4,4)
        
        R_h = np.eye(4)
        R_h[:3, :3] = R #(4,4)
        
        pts = p2_h @ R_h @ Tr_h  @ pts # (4, N*8)
        u =  pts[0, :] / np.clip(pts[2, :], 1e-6, None) 
        v =  pts[1, :] / np.clip(pts[2, :], 1e-6, None)
        pp_rectangles = []
        yolo_rectangles = []
        for i in range(0, u.size, 8):
            min_u = np.min(u[i:i+8])
            max_u = np.max(u[i:i+8])
            min_v = np.min(v[i:i+8])
            max_v = np.max(v[i:i+8])
            color = (0, 255, 255)
            pp_rectangles.append([int(min_u), int(min_v), int(max_u), int(max_v)])
            cv2.rectangle(img, (int(min_u), int(min_v)), (int(max_u), int(max_v)), color, 2)
            cv2.putText(img, f"PointPillars i:{i/8} {pp_scores[i//8]:.2f}", (int(min_u), int(min_v)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        i = 0
        for box in yolo_bb:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()   
            yolo_rectangles.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
            color = (255, 0, 255)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img, f"YOLOv11 i:{i} {conf:.2f}", (xyxy[2], xyxy[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            i+=1
        i = 0
        iou_matrix = []
        for pp_rect in pp_rectangles:
            row = []
            for yolo_rect in yolo_rectangles:
                iou = compute_iou(pp_rect, yolo_rect)
                i += 1
                row.append(iou)
            iou_matrix.append(row)
        return img, np.array(iou_matrix)




# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="point pillar checkpoint .pth")
    ap.add_argument("--bin", required=True, help="file KITTI .bin (x,y,z,intensity float32)")
    ap.add_argument("--pt", required=True, help="YOLOv11 checkpoint .pt")
    ap.add_argument("--img", required=True, help="file immagine KITTI png")
    ap.add_argument("--calib", required=True, default=None, help="file calibrazione KITTI")
    ap.add_argument("--score_thresh", type=float, default=SCORE_THRESH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None, help="salva la vista BEV in questo file PNG")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--open3d", action="store_true", help="vista 3D interattiva (serve open3d)")
    args = ap.parse_args()

    device = torch.device(args.device)
    pointpillar = load_model(args.ckpt, device)
    points = load_kitti_bin(args.bin)
    print(f"{len(points)} punti, device: {device}")

    boxes, scores, labels = detect(pointpillar, points, device, args.score_thresh)
    print(f"{len(boxes)} detection")
    for b, s, l in zip(boxes, scores, labels):
        print(f"{CLASS_NAMES[int(l) - 1]:11s} score={s:.2f} "
              f"xyz=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f}) lwh=({b[3]:.1f},{b[4]:.1f},{b[5]:.1f}) yaw={b[6]:.2f}")

    yolo = YOLO(args.pt).to(device)
    results = yolo(args.img, conf=0.5, classes=[0,3,5]) #0.351
    print(f"YOLOv11: {len(results[0].boxes)} detection")
    for box in results[0].boxes:
        print(f"YOLOv11: {box.cls.item()} score={box.conf.item():.2f} xyxy={box.xyxy[0].tolist()}")
    print("End of YOLOv11 detection")
    img = visualize_cam_projection(boxes, labels, *load_kitti_calib(args.calib))
    plt.figure(1).suptitle("PointPillars Projection on Camera Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    colors = {1: (255, 0, 255), 2: (0, 128, 230), 3: (255, 102, 0)}
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls.item())
        conf = box.conf.item()   
        color = colors.get(cls+1, (0, 0, 255))
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(img, f"YOLOv11 {conf:.2f}", (xyxy[0], xyxy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    plt.figure(2).suptitle("YOLOv11 Detection and PointPillars Projection on Camera Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img2, iou_matrix = visualize_pointpillar_yolo_fusion(results[0].boxes, boxes, scores, labels, *load_kitti_calib(args.calib))
    plt.figure(4).suptitle("PointPillars Bounding Boxes and YOLOv11 Bounding Boxes on Camera Image")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    rows, cols = linear_sum_assignment(-iou_matrix)
    matches = [(r, c, iou_matrix[r, c]) for r, c in zip(rows, cols) if iou_matrix[r, c] >= 0.5]
    pp_to_yolo = {r: (c, iou) for r, c, iou in matches}
    yolo_to_pp = {c: (r, iou) for r, c, iou in matches}
    
    fusion_fig = plt.figure(3)
    fusion_fig.suptitle("PointPillars + YOLOv11 Fusion")

    colors = {1: (0, 255, 0), 2: (255, 128, 25), 3: (0, 153, 255)}  # BGR
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]]
    P2, Tr, R = load_kitti_calib(args.calib)
    yolo_bb = results[0].boxes
    img_fusion = cv2.imread(args.img)
    if len(boxes) > 0:
        corners = boxes_to_corners_3d(boxes)                  # (N,8,3)
        n = corners.shape[0]
        pts_h = np.concatenate([corners.reshape(-1, 3), np.ones((n*8, 1))], axis=1).T

        p2_h = np.vstack([P2, [0, 0, 0, 1]])
        Tr_h = np.vstack([Tr, [0, 0, 0, 1]])
        R_h = np.eye(4); R_h[:3, :3] = R

        proj = p2_h @ R_h @ Tr_h @ pts_h                      # (4, N*8)
        z = np.clip(proj[2], 1e-6, None)
        uv = np.stack([proj[0] / z, proj[1] / z], axis=1).reshape(n, 8, 2)
        depth = proj[2].reshape(n, 8)

        for k in range(n):                                    # k = PointPillars box index
            if (depth[k] <= 0.1).any():                       # behind the camera
                continue
            p = uv[k]
            matched = k in pp_to_yolo
            col = colors.get(int(labels[k]), (0, 0, 255)) if matched else (0, 0, 255)  # red = PP only

            for a, b in edges:
                cv2.line(img_fusion, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]),
                        col, 2, cv2.LINE_AA)
            cv2.line(img_fusion, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[7]), col, 1, cv2.LINE_AA)
            cv2.line(img_fusion, tuple(int(v) for v in p[3]), tuple(int(v) for v in p[4]), col, 1, cv2.LINE_AA)

            depth_front = np.sort(depth[k])[:4].mean()        # first 4 corners (sorted in ascending order (from near to far)) and mean
            x, y = int(p[7][0]), int(p[7][1]) - 10            # where to put the text 
            if matched:
                c, s = pp_to_yolo[k]
                yb = yolo_bb[c]
                text = f"{CLASS_NAMES[int(yb.cls.item())]} {yb.conf.item():.2f} | {depth_front:.1f} m"
            else:
                text = f"PP only | {depth_front:.1f} m"
            cv2.putText(img_fusion, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    matched_yolo = {c for c, _ in pp_to_yolo.values()} 
    for c, bb in enumerate(yolo_bb):
        if c in matched_yolo:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb.xyxy[0].cpu().numpy()]
        name = results[0].names[int(bb.cls.item())]
        cv2.rectangle(img_fusion, (x1, y1), (x2, y2),(255, 0, 255), 2)
        cv2.putText(img_fusion, f"YOLO only: {name} {bb.conf.item():.2f}",
                    (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 1, cv2.LINE_AA)


    fusion_fig = plt.figure(3)
    fusion_fig.suptitle("PointPillars + YOLOv11 Fusion")
    plt.imshow(cv2.cvtColor(img_fusion, cv2.COLOR_BGR2RGB))       
    #plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    visualize_bev(points, boxes, scores, labels, save=args.save, show=not args.no_show)
    if args.open3d:
        visualize_open3d(points, boxes, labels)
    plt.show()
    plt.close("all")
