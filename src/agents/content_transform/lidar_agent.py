"""
LiDAR Agent with 3D Processing Pipeline
Uses computer vision techniques + BEV representations for semantic understanding
"""

from typing import Any, Dict, Optional, List, Tuple
from agents import BaseAgent
import numpy as np
import json
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import cv2
from io import BytesIO
import base64
from PIL import Image


@dataclass
class DetectedObject:
    """Represents a detected object from point cloud"""
    category: str
    position: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [length, width, height]
    num_points: int
    distance: float
    direction: str  # 'front', 'back', 'left', 'right', etc.
    confidence: float


class LiDARAgent(BaseAgent):
    """
    LiDAR processing agent that:
    1. Segments point cloud into objects using 3D clustering
    2. Classifies objects based on geometric features
    3. Generates rich BEV visualizations with semantic layers
    4. Extracts actionable semantic information
    5. Uses LLM only for final high-level interpretation
    """
    
    def __init__(self, client, model: str, agent_name: str):
        super().__init__(client, model, agent_name)
        
        # Clustering parameters
        self.dbscan_eps = 0.5  # meters
        self.dbscan_min_samples = 10
        
        # BEV visualization parameters
        self.bev_resolution = 800  # pixels
        self.bev_range = 50  # meters in each direction
    
    def process(self, point_cloud: np.ndarray, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process LiDAR point cloud with advanced 3D analysis
        
        Args:
            point_cloud: Nx4 array (x, y, z, intensity)
            context: Optional context from other agents
        """
        
        # Step 1: Preprocess point cloud
        processed_pc = self._preprocess_point_cloud(point_cloud)
        
        # Step 2: Ground segmentation (separate ground from objects)
        ground_points, object_points = self._segment_ground(processed_pc)
        
        # Step 3: 3D object detection via clustering
        detected_objects = self._detect_objects_3d(object_points)
        
        # Step 4: Generate rich BEV visualizations
        bev_images = self._generate_multi_layer_bev(
            ground_points, object_points
        )
        
        # Step 5: Extract semantic features
        semantic_features = self._extract_semantic_features(
            detected_objects, ground_points, object_points
        )
        
        # Step 6: Generate structured report (no LLM needed for most info)
        structured_report = self._generate_structured_report(
            semantic_features, detected_objects
        )
        
        # Step 7: Use LLM only for high-level scene interpretation
        scene_interpretation = self._generate_scene_interpretation(
            structured_report, bev_images, context
        )
        
        return {
            "agent": self.agent_name,
            "modality": "lidar",
            "detected_objects": [self._object_to_dict(obj) for obj in detected_objects],
            "semantic_features": semantic_features,
            "structured_report": structured_report,
            "observations": scene_interpretation,
            "bev_metadata": {
                "num_objects": len(detected_objects),
                "ground_points": len(ground_points),
                "object_points": len(object_points)
            }
        }
    
    def _preprocess_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Preprocess point cloud: remove outliers, filter by range"""
        # Remove points too far or too close
        distances = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        valid_mask = (distances > 1.0) & (distances < self.bev_range)
        
        # Remove points that are too high (e.g., birds, noise)
        valid_mask &= (pc[:, 2] < 5.0) & (pc[:, 2] > -3.0)
        
        return pc[valid_mask]
    
    def _segment_ground(self, pc: np.ndarray, 
                       ground_threshold: float = -1.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ground plane from objects using simple height thresholding
        
        Args:
            pc: Point cloud
            ground_threshold: Z-height below which points are likely ground
            
        Returns:
            (ground_points, object_points)
        """
        # Simple height-based segmentation (works well for flat urban scenes)
        
        ground_mask = pc[:, 2] < ground_threshold
        ground_points = pc[ground_mask]
        object_points = pc[~ground_mask]
        
        return ground_points, object_points
    
    def _detect_objects_3d(self, object_points: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using 3D clustering + vision-based classification
        
        Strategy:
        1. Cluster points into object candidates
        2. Generate individual object visualizations
        3. Use vision model to classify each cluster
        4. Return properly classified objects
        """
        if len(object_points) < self.dbscan_min_samples:
            return []
        
        # DBSCAN clustering in 3D space
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        ).fit(object_points[:, :3])
        
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        # Extract clusters
        clusters = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = object_points[cluster_mask]
            
            # Skip very small clusters
            if len(cluster_points) < 5:
                continue
            
            clusters.append(cluster_points)
        
        if not clusters:
            return []
                
        # Classify clusters using vision model
        detected_objects = self._classify_clusters_with_vision(clusters)
        
        return detected_objects
    
    def _classify_clusters_with_vision(self, clusters: List[np.ndarray]) -> List[DetectedObject]:
        """
        Classify multiple clusters using vision model
        
        Strategy: 
        1. Generate multi-view visualizations for each cluster (batch)
        2. Send to vision model for classification
        3. Parse classifications and create DetectedObject instances
        """
        detected_objects = []
        
        # Process clusters in batches to reduce API calls
        batch_size = 10  # Classify up to 10 objects per API call
        
        for batch_start in range(0, len(clusters), batch_size):
            batch_clusters = clusters[batch_start:batch_start + batch_size]
            
            # Generate visualizations for this batch
            cluster_visualizations = []
            cluster_metadata = []
            
            for i, cluster_points in enumerate(batch_clusters):
                # Extract geometric properties (for reference, not classification)
                min_coords = cluster_points[:, :3].min(axis=0)
                max_coords = cluster_points[:, :3].max(axis=0)
                dimensions = max_coords - min_coords
                center = (min_coords + max_coords) / 2
                distance = np.sqrt(center[0]**2 + center[1]**2)
                direction = self._get_direction(center[:2])
                
                # Generate multiple view angles for this cluster
                cluster_image = self._generate_cluster_visualization(cluster_points)
                cluster_visualizations.append(cluster_image)
                
                cluster_metadata.append({
                    'index': batch_start + i,
                    'center': center,
                    'dimensions': dimensions,
                    'distance': distance,
                    'direction': direction,
                    'num_points': len(cluster_points)
                })
            
            # Classify batch with vision model
            classifications = self._classify_batch_with_llm(
                cluster_visualizations, 
                cluster_metadata
            )
            
            # Create DetectedObject instances
            for metadata, classification in zip(cluster_metadata, classifications):
                if classification['category'] != 'unknown' and classification['confidence'] > 0.3:
                    detected_objects.append(DetectedObject(
                        category=classification['category'],
                        position=metadata['center'],
                        dimensions=metadata['dimensions'],
                        num_points=metadata['num_points'],
                        distance=metadata['distance'],
                        direction=metadata['direction'],
                        confidence=classification['confidence']
                    ))
        
        return detected_objects
    
    def _generate_cluster_visualization(self, points: np.ndarray, 
                                       img_size: int = 256) -> np.ndarray:
        """
        Generate multi-view visualization of a single object cluster
        
        Creates a 2x2 grid showing:
        - Top view (XY plane)
        - Side view (XZ plane)
        - Front view (YZ plane)
        - 3D perspective
        
        This gives the vision model enough context to classify the object
        """
        # Center and normalize the cluster
        center = points[:, :3].mean(axis=0)
        centered = points[:, :3] - center
        
        # Calculate bounds for consistent scaling
        max_range = max(
            centered[:, 0].max() - centered[:, 0].min(),
            centered[:, 1].max() - centered[:, 1].min(),
            centered[:, 2].max() - centered[:, 2].min()
        )
        scale = (img_size * 0.35) / max_range if max_range > 0 else 1
        
        # Create 2x2 grid
        grid = np.ones((img_size * 2, img_size * 2, 3), dtype=np.uint8) * 255
        
        # Helper to draw points on a 2D projection
        def draw_projection(ax1, ax2, quadrant_x, quadrant_y, title):
            view_size = img_size
            offset_x = quadrant_x * img_size
            offset_y = quadrant_y * img_size
            
            # Project points
            proj_x = (centered[:, ax1] * scale + view_size / 2).astype(int)
            proj_y = (centered[:, ax2] * scale + view_size / 2).astype(int)
            
            # Clip to bounds
            valid = (proj_x >= 0) & (proj_x < view_size) & (proj_y >= 0) & (proj_y < view_size)
            proj_x = proj_x[valid]
            proj_y = proj_y[valid]
            
            # Use intensity for color if available
            if points.shape[1] > 3:
                intensities = points[valid, 3]
                intensities = ((intensities - intensities.min()) / 
                              (intensities.max() - intensities.min() + 1e-6) * 255).astype(np.uint8)
            else:
                intensities = np.ones(len(proj_x), dtype=np.uint8) * 200
            
            # Draw points
            for x, y, intensity in zip(proj_x, proj_y, intensities):
                color = (int(intensity), int(intensity), int(intensity))
                cv2.circle(grid, 
                          (offset_x + x, offset_y + view_size - y - 1),
                          2, color, -1)
            
            # Add axes and title
            # X axis (red)
            cv2.line(grid, 
                    (offset_x + view_size // 2, offset_y + view_size // 2),
                    (offset_x + view_size // 2 + 30, offset_y + view_size // 2),
                    (0, 0, 255), 2)
            # Y axis (green)
            cv2.line(grid,
                    (offset_x + view_size // 2, offset_y + view_size // 2),
                    (offset_x + view_size // 2, offset_y + view_size // 2 - 30),
                    (0, 255, 0), 2)
            
            # Title
            cv2.putText(grid, title, (offset_x + 10, offset_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Generate views
        draw_projection(0, 1, 0, 0, "Top (XY)")      # Top-left: XY plane
        draw_projection(0, 2, 1, 0, "Side (XZ)")     # Top-right: XZ plane  
        draw_projection(1, 2, 0, 1, "Front (YZ)")    # Bottom-left: YZ plane
        
        # Bottom-right: 3D-ish isometric view
        # Rotate points for isometric projection
        angle = np.pi / 6  # 30 degrees
        rot_x = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
        rot_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
        rotated = centered @ rot_x.T @ rot_y.T
        
        iso_x = ((rotated[:, 0] + rotated[:, 1] * 0.5) * scale + img_size / 2).astype(int)
        iso_y = ((rotated[:, 2] - rotated[:, 1] * 0.5) * scale + img_size / 2).astype(int)
        
        offset_x = img_size
        offset_y = img_size
        valid = (iso_x >= 0) & (iso_x < img_size) & (iso_y >= 0) & (iso_y < img_size)
        iso_x = iso_x[valid]
        iso_y = iso_y[valid]
        
        if points.shape[1] > 3:
            intensities = points[valid, 3]
            intensities = ((intensities - intensities.min()) / 
                          (intensities.max() - intensities.min() + 1e-6) * 255).astype(np.uint8)
        else:
            intensities = np.ones(len(iso_x), dtype=np.uint8) * 200
        
        for x, y, intensity in zip(iso_x, iso_y, intensities):
            color = (int(intensity), int(intensity), int(intensity))
            cv2.circle(grid,
                      (offset_x + x, offset_y + img_size - y - 1),
                      2, color, -1)
        
        cv2.putText(grid, "3D View", (offset_x + 10, offset_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return grid
    
    def _classify_batch_with_llm(self, 
                                cluster_images: List[np.ndarray],
                                cluster_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Classify a batch of object clusters using vision model
        
        Sends multiple cluster visualizations at once for efficiency
        """
        # Combine images into a single visualization
        if len(cluster_images) == 1:
            combined_image = cluster_images[0]
        else:
            # Arrange in a grid
            n_clusters = len(cluster_images)
            cols = min(3, n_clusters)
            rows = (n_clusters + cols - 1) // cols
            
            h, w = cluster_images[0].shape[:2]
            combined_image = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
            
            for idx, img in enumerate(cluster_images):
                row = idx // cols
                col = idx % cols
                combined_image[row*h:(row+1)*h, col*w:(col+1)*w] = img
                
                # Add cluster number
                cv2.putText(combined_image, f"#{idx}", 
                           (col*w + 10, row*h + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        # Convert to base64
        b64_image = self._image_to_base64(combined_image)
        
        # Build metadata summary
        metadata_text = "Object Cluster Details:\n"
        for i, meta in enumerate(cluster_metadata):
            metadata_text += f"\nCluster #{i}:\n"
            metadata_text += f"  Position: ({meta['center'][0]:.1f}, {meta['center'][1]:.1f}, {meta['center'][2]:.1f})m\n"
            metadata_text += f"  Size: {meta['dimensions'][0]:.1f} x {meta['dimensions'][1]:.1f} x {meta['dimensions'][2]:.1f}m (L x W x H)\n"
            metadata_text += f"  Distance: {meta['distance']:.1f}m\n"
            metadata_text += f"  Direction: {meta['direction']}\n"
            metadata_text += f"  Points: {meta['num_points']}\n"
        
        system_prompt = """You are an expert in 3D object classification for autonomous driving.

You will see multi-view visualizations of objects detected from LiDAR point clouds. Each object is shown in 4 views:
- Top view (XY): Looking down at the object
- Side view (XZ): Looking from the side
- Front view (YZ): Looking from the front
- 3D view: Isometric perspective

Your task: Classify each object into one of these categories:
- car: Standard passenger vehicles
- truck: Large vehicles (pickups, delivery trucks, semi-trucks)
- bus: Buses and large passenger vehicles
- pedestrian: People, adults or children
- bicycle: Bicycles, e-bikes
- motorcycle: Motorcycles, scooters
- trailer: Trailers, mobile structures
- barrier: Concrete barriers, Jersey barriers
- traffic_cone: Traffic cones, road markers
- construction_vehicle: Construction equipment, machinery
- unknown: Cannot determine

Guidelines:
- Use shape, size, and point density to classify
- Cars are roughly rectangular, 4-5m long, 1.6-2m wide
- Pedestrians are small, vertical, roughly cylindrical
- Trucks/buses are larger versions of cars (>5m)
- Bicycles are thin, elongated (~2m long, <1m wide)
- Consider the metadata (dimensions, point count) as additional context
- If unsure, use 'unknown'

Output format (JSON):
{
  "classifications": [
    {"cluster": 0, "category": "car", "confidence": 0.9, "reasoning": "rectangular shape, ~4.5m long"},
    {"cluster": 1, "category": "pedestrian", "confidence": 0.85, "reasoning": "small, vertical, ~1.7m tall"},
    ...
  ]
}

Be precise and provide confidence scores (0.0-1.0)."""

        user_prompt = f"""Classify these {len(cluster_images)} object cluster(s):

{metadata_text}

Analyze the multi-view visualizations and metadata, then classify each object.
Output valid JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]}
        ]
        
        response = self.call_llm(messages, temperature=0.2)
        
        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            classifications_list = result.get('classifications', [])
            
            # Convert to list matching cluster order
            classifications = []
            for i in range(len(cluster_metadata)):
                # Find matching classification
                cluster_result = next(
                    (c for c in classifications_list if c.get('cluster') == i),
                    {'category': 'unknown', 'confidence': 0.5}
                )
                classifications.append({
                    'category': cluster_result.get('category', 'unknown'),
                    'confidence': cluster_result.get('confidence', 0.5)
                })
            
            return classifications
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [LiDAR] Warning: Failed to parse classifications: {e}")
            # Return default classifications
            return [{'category': 'unknown', 'confidence': 0.5} 
                   for _ in cluster_metadata]
    
    def _get_direction(self, position_2d: np.ndarray) -> str:
        """Get direction label from 2D position relative to ego vehicle"""
        x, y = position_2d
        angle = np.arctan2(y, x) * 180 / np.pi
        
        # Normalize to [0, 360)
        angle = (angle + 360) % 360
        
        # 8-directional classification
        if 337.5 <= angle or angle < 22.5:
            return "front_right"
        elif 22.5 <= angle < 67.5:
            return "front"
        elif 67.5 <= angle < 112.5:
            return "front_left"
        elif 112.5 <= angle < 157.5:
            return "left"
        elif 157.5 <= angle < 202.5:
            return "back_left"
        elif 202.5 <= angle < 247.5:
            return "back"
        elif 247.5 <= angle < 292.5:
            return "back_right"
        else:
            return "right"
    
    def _generate_multi_layer_bev(self, 
                                  ground_points: np.ndarray,
                                  object_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate multi-layer BEV visualization from point cloud data
        Returns density and height maps for LLM interpretation
        """
        res = self.bev_resolution
        r = self.bev_range
        
        # Initialize layers
        height_map = np.zeros((res, res), dtype=np.float32)
        density_map = np.zeros((res, res), dtype=np.float32)
        
        # Helper function to convert coordinates to pixels
        def to_pixels(coords):
            x_pix = ((coords[:, 0] + r) / (2 * r) * res).astype(int)
            y_pix = ((coords[:, 1] + r) / (2 * r) * res).astype(int)
            x_pix = np.clip(x_pix, 0, res - 1)
            y_pix = np.clip(y_pix, 0, res - 1)
            return x_pix, y_pix
        
        # Fill density map with all points
        all_points = np.vstack([ground_points, object_points])
        x_pix, y_pix = to_pixels(all_points)
        
        for x, y, z in zip(x_pix, y_pix, all_points[:, 2]):
            density_map[y, x] += 1
            height_map[y, x] = max(height_map[y, x], z)
        
        # Normalize density map with log scale
        density_map = np.log1p(density_map)
        density_map = (density_map / density_map.max() * 255).astype(np.uint8) if density_map.max() > 0 else density_map.astype(np.uint8)
        
        # Create rich visualization map
        visualization_map = np.zeros((res, res, 3), dtype=np.uint8)
        
        # Ground points in dark blue
        ground_x, ground_y = to_pixels(ground_points)
        for x, y in zip(ground_x, ground_y):
            visualization_map[y, x] = [80, 80, 120]  # Dark blue for ground
        
        # Object points in hot colormap (yellow to red based on height)
        obj_x, obj_y = to_pixels(object_points)
        obj_heights = object_points[:, 2]
        
        # Normalize heights to 0-1
        if len(obj_heights) > 0 and obj_heights.max() > obj_heights.min():
            norm_heights = (obj_heights - obj_heights.min()) / (obj_heights.max() - obj_heights.min())
        else:
            norm_heights = np.ones(len(obj_heights)) * 0.5
        
        for x, y, height_norm in zip(obj_x, obj_y, norm_heights):
            # Hot colormap: low height = yellow, high height = red
            if height_norm < 0.5:
                # Yellow to orange
                r_val = 255
                g_val = int(255 * (1 - height_norm * 2))
                b_val = 0
            else:
                # Orange to red
                r_val = 255
                g_val = int(255 * (1 - (height_norm - 0.5) * 2))
                b_val = 0
            
            visualization_map[y, x] = [b_val, g_val, r_val]  # BGR format
        
        # Add ego vehicle marker (bright green cross)
        center = res // 2
        marker_size = 15
        cv2.line(visualization_map, 
                (center - marker_size, center), 
                (center + marker_size, center),
                (0, 255, 0), 3)
        cv2.line(visualization_map,
                (center, center - marker_size),
                (center, center + marker_size),
                (0, 255, 0), 3)
        
        # Flip y-axis so forward is up
        visualization_map = cv2.flip(visualization_map, 0)
        height_map = cv2.flip(height_map, 0)
        density_map = cv2.flip(density_map, 0)
        
        # Add distance circles (10m, 20m, 30m, 40m)
        for dist in [10, 20, 30, 40]:
            radius = int(dist / (2 * r) * res)
            cv2.circle(visualization_map, (center, center), radius, (100, 100, 100), 1)
            # Add distance label
            label_pos = (center + 5, center - radius + 15)
            cv2.putText(visualization_map, f"{dist}m", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Add cardinal directions
        text_offset = int(res * 0.45)
        cv2.putText(visualization_map, "FRONT", (center - 25, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(visualization_map, "BACK", (center - 20, res - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(visualization_map, "L", (10, center + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(visualization_map, "R", (res - 20, center + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # self._debug_plot_bev_layers(visualization_map, height_map, density_map)
        
        return {
            'semantic': visualization_map,  # Renamed but same purpose - visual representation
            'height': height_map,
            'density': density_map
        }
    
    def _extract_semantic_features(self,
                                   detected_objects: List[DetectedObject],
                                   ground_points: np.ndarray,
                                   object_points: np.ndarray) -> Dict[str, Any]:
        """Extract high-level semantic features from processed data"""
        
        # Count objects by category
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj.category] = object_counts.get(obj.category, 0) + 1
        
        # Spatial distribution of objects
        objects_by_direction = {
            'front': [], 'back': [], 'left': [], 'right': [],
            'front_left': [], 'front_right': [], 'back_left': [], 'back_right': []
        }
        
        for obj in detected_objects:
            objects_by_direction[obj.direction].append(obj)
        
        # Distance-based grouping
        close_objects = [obj for obj in detected_objects if obj.distance < 10]
        medium_objects = [obj for obj in detected_objects if 10 <= obj.distance < 30]
        far_objects = [obj for obj in detected_objects if obj.distance >= 30]
        
        # Scene density analysis
        total_points = len(ground_points) + len(object_points)
        object_ratio = len(object_points) / total_points if total_points > 0 else 0
        
        # Traffic density indicator
        vehicles = [obj for obj in detected_objects if obj.category in ['car', 'truck', 'bus']]
        traffic_density = "heavy" if len(vehicles) > 10 else "moderate" if len(vehicles) > 5 else "light"
        
        return {
            'total_objects': len(detected_objects),
            'object_counts': object_counts,
            'objects_by_direction': {
                dir: len(objs) for dir, objs in objects_by_direction.items()
            },
            'distance_distribution': {
                'close': len(close_objects),
                'medium': len(medium_objects),
                'far': len(far_objects)
            },
            'scene_characteristics': {
                'object_point_ratio': float(object_ratio),
                'traffic_density': traffic_density,
                'total_points': total_points
            },
            'nearest_object': min(detected_objects, key=lambda x: x.distance) if detected_objects else None
        }
    
    def _generate_structured_report(self,
                                    semantic_features: Dict[str, Any],
                                    detected_objects: List[DetectedObject]) -> str:
        """Generate structured text report from semantic features"""
        
        report_lines = []
        
        # Summary
        report_lines.append("=== LiDAR Scene Analysis ===\n")
        report_lines.append(f"Total detected objects: {semantic_features['total_objects']}")
        
        # Object counts
        if semantic_features['object_counts']:
            report_lines.append("\nObject Distribution:")
            for category, count in sorted(semantic_features['object_counts'].items()):
                report_lines.append(f"  - {count} {category}(s)")
        
        # Spatial distribution
        report_lines.append("\nSpatial Distribution:")
        for direction, count in semantic_features['objects_by_direction'].items():
            if count > 0:
                objs_in_dir = [obj for obj in detected_objects if obj.direction == direction]
                categories = ', '.join(set(obj.category for obj in objs_in_dir))
                report_lines.append(f"  - {direction}: {count} objects ({categories})")
        
        # Distance-based information
        dist_dist = semantic_features['distance_distribution']
        report_lines.append("\nDistance Distribution:")
        report_lines.append(f"  - Close (<10m): {dist_dist['close']} objects")
        report_lines.append(f"  - Medium (10-30m): {dist_dist['medium']} objects")
        report_lines.append(f"  - Far (>30m): {dist_dist['far']} objects")
        
        # Nearest object
        if semantic_features['nearest_object']:
            nearest = semantic_features['nearest_object']
            report_lines.append(f"\nNearest Object:")
            report_lines.append(f"  - Type: {nearest.category}")
            report_lines.append(f"  - Distance: {nearest.distance:.1f}m")
            report_lines.append(f"  - Direction: {nearest.direction}")
        
        # Scene characteristics
        scene = semantic_features['scene_characteristics']
        report_lines.append(f"\nScene Characteristics:")
        report_lines.append(f"  - Traffic density: {scene['traffic_density']}")
        report_lines.append(f"  - Object point ratio: {scene['object_point_ratio']:.2%}")
        
        return '\n'.join(report_lines)
    
    def _generate_scene_interpretation(self,
                                      structured_report: str,
                                      bev_images: Dict[str, np.ndarray],
                                      context: Optional[Dict]) -> str:
        """
        Use LLM for high-level scene interpretation only
        LLM receives structured data + BEV visualization
        """
        
        # Convert semantic BEV to base64
        bev_semantic = bev_images['semantic']
        b64_bev = self._image_to_base64(bev_semantic)
        
        system_prompt = """You are an autonomous driving scene understanding expert.

You receive:
1. A structured LiDAR analysis report with detected objects
2. A Bird's Eye View visualization showing object locations

The BEV image shows:
- Green cross: ego vehicle position
- Colored boxes: detected objects (with labels)
- Gray circles: distance markers (10m, 20m, 30m, 40m)
- Colors: Green=cars, Orange=trucks, Magenta=buses, Yellow=pedestrians, Cyan=bicycles, Gray=barriers, Orange-red=traffic cones, Light blue=unknown objects

Your task:
- Provide high-level scene interpretation
- Identify potential risks or notable situations
- Describe the overall driving context
- Note any patterns or important spatial relationships

Be concise and focus on actionable insights for autonomous driving."""

        user_prompt = f"""Analyze this driving scene from LiDAR data:

{structured_report}

Provide a high-level interpretation of the scene, including:
1. Overall scene context (urban/highway, crowded/sparse, etc.)
2. Key objects and their significance
3. Potential risks or safety concerns
4. Notable spatial patterns or relationships"""

        if context:
            user_prompt += f"\n\nAdditional context from other sensors:\n{json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_bev}"}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]}
        ]
        
        return self.call_llm(messages, temperature=0.4)
    
    @staticmethod
    def _object_to_dict(obj: DetectedObject) -> Dict[str, Any]:
        """Convert DetectedObject to dictionary"""
        return {
            'category': obj.category,
            'position': obj.position.tolist(),
            'dimensions': obj.dimensions.tolist(),
            'num_points': obj.num_points,
            'distance': float(obj.distance),
            'direction': obj.direction,
            'confidence': float(obj.confidence)
        }
    
    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _debug_plot_bev_layers(self, semantic_map, height_map, density_map):
        """Debug function to plot BEV layers"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Semantic BEV")
        plt.imshow(semantic_map)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Height Map")
        plt.imshow(height_map, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Density Map")
        plt.imshow(density_map, cmap='hot')
        plt.axis('off')
        
        plt.show()


# Factory function
def create_lidar_agent(client, model: str):
    """Create LiDAR agent"""
    return LiDARAgent(client, model, "LiDARAgent")
