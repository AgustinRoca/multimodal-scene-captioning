from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("Warning: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")


class NuScenesLoader:
    """Loads and preprocesses nuScenes data for the captioning pipeline"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        """
        Initialize nuScenes loader
        
        Args:
            dataroot: Path to nuScenes dataset root directory
            version: Dataset version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
        """
        if not NUSCENES_AVAILABLE:
            raise ImportError("nuscenes-devkit is required. Install with: pip install nuscenes-devkit")
        
        self.dataroot = Path(dataroot)
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
        
        # Camera names in nuScenes
        self.camera_channels = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT', 
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
    
    def get_scene_list(self) -> List[Dict]:
        """Get list of all scenes with metadata"""
        scenes = []
        for scene in self.nusc.scene:
            scenes.append({
                'token': scene['token'],
                'name': scene['name'],
                'description': scene['description'],
                'nbr_samples': scene['nbr_samples'],
                'first_sample_token': scene['first_sample_token']
            })
        return scenes
    
    def load_sample(self, sample_token: str) -> Dict:
        """
        Load complete sample data
        
        Args:
            sample_token: nuScenes sample token
            
        Returns:
            Dictionary containing all sensor data and annotations
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Load camera images
        images = []
        camera_names = []
        for cam_channel in self.camera_channels:
            if cam_channel in sample['data']:
                cam_token = sample['data'][cam_channel]
                img, cam_name = self._load_camera(cam_token)
                images.append(img)
                camera_names.append(cam_name)
        
        # Load LiDAR point cloud
        lidar_token = sample['data']['LIDAR_TOP']
        point_cloud = self._load_lidar(lidar_token)
        
        # Load annotations
        annotations = self._load_annotations(sample['anns'])
        
        # Get scene metadata
        scene = self.nusc.get('scene', sample['scene_token'])
        
        return {
            'sample_token': sample_token,
            'timestamp': sample['timestamp'],
            'scene_description': scene['description'],
            'scene_name': scene['name'],
            'images': images,
            'camera_names': camera_names,
            'point_cloud': point_cloud,
            'annotations': annotations,
            'metadata': {
                'location': self.nusc.get('log', scene['log_token'])['location'],
                'nbr_objects': len(annotations)
            }
        }
    
    def load_scene_samples(self, scene_token: str, 
                          max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load all samples from a scene
        
        Args:
            scene_token: Scene token
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of sample dictionaries
        """
        scene = self.nusc.get('scene', scene_token)
        
        # Get first sample
        sample_token = scene['first_sample_token']
        samples = []
        count = 0
        
        while sample_token != '':
            if max_samples and count >= max_samples:
                break
            
            sample_data = self.load_sample(sample_token)
            samples.append(sample_data)
            
            # Get next sample
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            count += 1
        
        return samples
    
    def _load_camera(self, cam_token: str) -> Tuple[np.ndarray, str]:
        """Load camera image"""
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = self.dataroot / cam_data['filename']
        
        img = Image.open(img_path)
        img_array = np.array(img)
        
        return img_array, cam_data['channel']
    
    def _load_lidar(self, lidar_token: str) -> np.ndarray:
        """Load LiDAR point cloud"""
        lidar_data = self.nusc.get('sample_data', lidar_token)
        pcl_path = self.dataroot / lidar_data['filename']
        
        # Load point cloud
        pc = LidarPointCloud.from_file(str(pcl_path))
        
        # Convert to numpy array (4 x N) -> (N x 4)
        points = pc.points.T  # Now shape is (N, 4) with columns [x, y, z, intensity]
        
        return points
    
    def _load_annotations(self, ann_tokens: List[str]) -> List[Dict]:
        """Load object annotations"""
        annotations = []
        
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get category
            category = ann['category_name']
            
            # Get attributes
            attributes = []
            for attr_token in ann['attribute_tokens']:
                attr = self.nusc.get('attribute', attr_token)
                attributes.append(attr['name'])
            
            # Get visibility
            visibility_token = ann['visibility_token']
            visibility = self.nusc.get('visibility', visibility_token)
            
            annotation_dict = {
                'token': ann_token,
                'category_name': category,
                'instance_token': ann['instance_token'],
                'translation': ann['translation'],  # [x, y, z] in global frame
                'size': ann['size'],  # [width, length, height]
                'rotation': ann['rotation'],  # quaternion
                'velocity': self.nusc.box_velocity(ann_token),  # [vx, vy]
                'attribute_tokens': attributes,
                'visibility_token': visibility['description'],
                'num_lidar_pts': ann['num_lidar_pts'],
                'num_radar_pts': ann['num_radar_pts']
            }
            
            annotations.append(annotation_dict)
        
        return annotations
    
    def get_sample_by_scene_index(self, scene_idx: int, sample_idx: int = 0) -> Dict:
        """
        Load sample by scene index and sample index within scene
        
        Args:
            scene_idx: Index of scene in dataset
            sample_idx: Index of sample within scene (0 for first sample)
        """
        scene = self.nusc.scene[scene_idx]
        samples = self.load_scene_samples(scene['token'], max_samples=sample_idx + 1)
        return samples[sample_idx] if samples else None


class MockNuScenesLoader:
    """Mock loader for testing without actual nuScenes data"""
    
    def __init__(self, dataroot: str = None, version: str = 'v1.0-mini'):
        print("Warning: Using MockNuScenesLoader - no actual data will be loaded")
        self.camera_channels = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT', 
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
    
    def get_scene_list(self) -> List[Dict]:
        """Return mock scene list"""
        return [
            {
                'token': 'mock_scene_001',
                'name': 'scene-0001',
                'description': 'Mock scene with vehicles at intersection',
                'nbr_samples': 10,
                'first_sample_token': 'mock_sample_001'
            }
        ]
    
    def load_sample(self, sample_token: str) -> Dict:
        """Generate mock sample data"""
        # Generate random images
        images = [
            np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8) 
            for _ in range(6)
        ]
        
        # Generate random point cloud
        point_cloud = np.random.randn(10000, 4).astype(np.float32)
        
        # Generate mock annotations
        annotations = [
            {
                'token': 'mock_ann_001',
                'category_name': 'vehicle.car',
                'translation': [10.0, 2.0, 0.5],
                'size': [2.0, 4.5, 1.5],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'velocity': [3.0, 0.5],
                'attribute_tokens': ['vehicle.moving'],
                'visibility_token': '60-80% visibility',
                'num_lidar_pts': 150,
                'num_radar_pts': 5
            },
            {
                'token': 'mock_ann_002',
                'category_name': 'human.pedestrian.adult',
                'translation': [8.0, -3.0, 1.0],
                'size': [0.5, 0.5, 1.8],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'velocity': [0.5, 0.2],
                'attribute_tokens': ['pedestrian.moving'],
                'visibility_token': '80-100% visibility',
                'num_lidar_pts': 80,
                'num_radar_pts': 0
            }
        ]
        
        return {
            'sample_token': sample_token,
            'timestamp': 1532402927647951,
            'scene_description': 'Mock driving scene',
            'scene_name': 'scene-0001',
            'images': images,
            'camera_names': self.camera_channels,
            'point_cloud': point_cloud,
            'annotations': annotations,
            'metadata': {
                'location': 'boston-seaport',
                'nbr_objects': len(annotations)
            }
        }
    
    def load_scene_samples(self, scene_token: str, 
                          max_samples: Optional[int] = None) -> List[Dict]:
        """Generate mock scene samples"""
        n_samples = min(max_samples or 5, 5)
        return [self.load_sample(f'mock_sample_{i:03d}') for i in range(n_samples)]
    
    def get_sample_by_scene_index(self, scene_idx: int, sample_idx: int = 0) -> Dict:
        """Get mock sample by indices"""
        return self.load_sample(f'mock_sample_{sample_idx:03d}')


def create_loader(dataroot: Optional[str] = None, version: str = 'v1.0-mini', 
                 use_mock: bool = False) -> NuScenesLoader:
    """
    Factory function to create appropriate loader
    
    Args:
        dataroot: Path to nuScenes dataset
        version: Dataset version
        use_mock: If True, use mock loader even if nuScenes is available
    """
    if use_mock or not NUSCENES_AVAILABLE or dataroot is None:
        return MockNuScenesLoader(dataroot, version)
    else:
        return NuScenesLoader(dataroot, version)
