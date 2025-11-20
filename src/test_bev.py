import torch
from mmcv import Config
from mmdet3d.models import build_model
from nuscenes.nuscenes import NuScenes
import cv2

# -----------------------------------------------------------
# 1. Load the BEVFusion segmentation config (inline minimal)
# -----------------------------------------------------------
cfg = Config.fromstring("""
model = dict(
    type='BEVFusion',
    use_lidar=True,
    use_camera=True,
    fusion_method='concat',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=[0.2, 0.2, 8.0], max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder', in_channels=5, sparse_shape=[41, 1600, 1408], output_channels=128),
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2]),
    bev_encoder=dict(
        type='BEVEncoder',
        in_channels=256,
        out_channels=256),
    head=dict(
        type='BEVOccHead',
        in_channels=256,
        out_channels=16)  # 16 BEV semantic classes
)
""", ".py")

# -----------------------------------------------------------
# 2. Build the model
# -----------------------------------------------------------
model = build_model(cfg.model, train_cfg=None, test_cfg=None)
model.cuda()
model.eval()

# -----------------------------------------------------------
# 3. Load checkpoint weights
# -----------------------------------------------------------
ckpt = torch.load("models/bevfusion-seg.pth", map_location="cpu")
model.load_state_dict(ckpt['state_dict'], strict=False)
print("Loaded checkpoint.")

# -----------------------------------------------------------
# 4. Load nuScenes mini
# -----------------------------------------------------------
nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes/v1.0-mini", verbose=False)

sample = nusc.sample[0]
lidar_token = sample['data']['LIDAR_TOP']
cam_front_token = sample['data']['CAM_FRONT']

lidar_data = nusc.get_sample_data(lidar_token, use_flat=True)
cam_data = nusc.get_sample_data(cam_front_token)

# -----------------------------------------------------------
# 5. Preprocess data (simplified)
#     BEVFusion expects:
#       - images transformed + normalized
#       - lidar in numpy array format (x,y,z,r)
# -----------------------------------------------------------

img = cv2.imread(cam_data['filename'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1600, 900))
img = torch.tensor(img).permute(2,0,1).float().cuda().unsqueeze(0)

points = lidar_data[1][:, :5]  # x,y,z,intensity,ring
points = torch.tensor(points).float().cuda().unsqueeze(0)

# -----------------------------------------------------------
# 6. Forward pass
# -----------------------------------------------------------
with torch.no_grad():
    output = model(img=img, points=points)

bev_seg = output['bev_seg']   # shape: (1, C, H, W)

print("BEV segmentation shape:", bev_seg.shape)
