from typing import Any, Dict, Optional
from agents import BaseAgent
import numpy as np
import json

class LiDARAgent(BaseAgent):
    """Processes LiDAR point cloud data"""
    
    def process(self, point_cloud: np.ndarray, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process LiDAR point cloud
        
        Args:
            point_cloud: Nx4 array (x, y, z, intensity)
            context: Optional context from other agents
        """
        # Generate BEV (Bird's Eye View) representation
        bev_description = self._analyze_point_cloud(point_cloud)
        
        system_prompt = """You are a LiDAR data expert analyzing 3D point clouds from a self-driving car.
Interpret the spatial information focusing on:
- 3D object positions and distances
- Object dimensions and shapes
- Spatial relationships between objects
- Occluded or partially visible objects
- Depth information for scene understanding

Provide structured, precise observations."""

        user_prompt = f"""Analyze this LiDAR point cloud data:

Point Cloud Statistics:
{bev_description}

Provide insights about the 3D scene structure and object spatial relationships."""

        if context:
            user_prompt += f"\n\nContext from other sensors:\n{json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.3)
        
        return {
            "agent": self.agent_name,
            "modality": "lidar",
            "point_cloud_stats": bev_description,
            "observations": response
        }
    
    @staticmethod
    def _analyze_point_cloud(pc: np.ndarray) -> Dict[str, Any]:
        """Extract statistical features from point cloud"""
        return {
            "num_points": len(pc),
            "spatial_extent": {
                "x_range": [float(pc[:, 0].min()), float(pc[:, 0].max())],
                "y_range": [float(pc[:, 1].min()), float(pc[:, 1].max())],
                "z_range": [float(pc[:, 2].min()), float(pc[:, 2].max())]
            },
            "density_clusters": int(len(pc) / 1000),  # Simplified
            "mean_intensity": float(pc[:, 3].mean()) if pc.shape[1] > 3 else None
        }