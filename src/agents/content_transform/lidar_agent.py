"""
Specialized LiDAR Agent with multiple processing backends
Supports: GPT-4o, Qwen2-VL, and traditional point cloud analysis
"""

from typing import Any, Dict, Optional, Literal
from agents import BaseLLMAgent
import numpy as np
import json
from io import BytesIO
import base64
from PIL import Image

class LiDARAgent(BaseLLMAgent):
    """Processes LiDAR point cloud data with specialized backends"""
    
    def __init__(self, client, model: str, agent_name: str, 
                 backend: Literal["llm", "qwen", "hybrid"] = "hybrid"):
        """
        Initialize LiDAR agent with specialized backend
        
        Args:
            client: Azure OpenAI client
            model: Model name (e.g., 'gpt-4o-mini', 'qwen-vl')
            agent_name: Name of the agent
            backend: Processing backend
                - "llm": Use LLM with statistics only (cheap, fast)
                - "qwen" (not implemented): Use Qwen2-VL with BEV image (better for spatial understanding)
                - "hybrid": Use both and combine insights (best quality)
        """
        super().__init__(client, model, agent_name)
        self.backend = backend

        if backend == "qwen":
            raise NotImplementedError("Qwen2-VL backend is not implemented in this version.")
        
        # For Qwen or hybrid mode, we'll generate BEV visualizations
        self.use_visualization = backend in ["qwen", "hybrid"]
    
    def process(self, point_cloud: np.ndarray, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process LiDAR point cloud with specialized backend
        
        Args:
            point_cloud: Nx4 array (x, y, z, intensity)
            context: Optional context from other agents
        """
        # Generate statistical analysis
        bev_description = self._analyze_point_cloud(point_cloud)
        
        result = {
            "agent": self.agent_name,
            "modality": "lidar",
            "point_cloud_stats": bev_description,
        }
        
        if self.backend == "llm":
            # Traditional LLM processing with statistics
            result["observations"] = self._process_with_llm(bev_description, context)
            
        # elif self.backend == "qwen":
        #     # Use Qwen2-VL with BEV visualization
        #     bev_image = self._generate_bev_visualization(point_cloud)
        #     result["observations"] = self._process_with_vision_model(
        #         bev_image, bev_description, context
        #     )
            
        elif self.backend == "hybrid":
            # Combine both approaches
            bev_image = self._generate_bev_visualization(point_cloud)
            
            # Get insights from both
            llm_insights = self._process_with_llm(bev_description, context)
            vision_insights = self._process_with_vision_model(
                bev_image, bev_description, context
            )
            
            # Combine insights
            result["observations"] = self._combine_insights(
                llm_insights, vision_insights, bev_description
            )
        
        return result
    
    def _process_with_llm(self, bev_description: Dict, context: Optional[Dict]) -> str:
        """Process using traditional LLM with statistics only"""
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
{json.dumps(bev_description, indent=2)}

Provide insights about the 3D scene structure and object spatial relationships."""

        if context:
            user_prompt += f"\n\nContext from other sensors:\n{json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_llm(messages, temperature=0.3)
    
    def _process_with_vision_model(self, bev_image: np.ndarray, 
                                   bev_description: Dict,
                                   context: Optional[Dict]) -> str:
        """Process using vision-language model with BEV visualization"""
        
        # Convert BEV image to base64
        b64_image = self._image_to_base64(bev_image)
        
        system_prompt = """You are a LiDAR visualization expert analyzing Bird's Eye View (BEV) representations of 3D point clouds.

The BEV image shows:
- Color intensity represents point density (brighter = more points)
- X-axis: left-right position relative to ego vehicle
- Y-axis: forward-backward position relative to ego vehicle
- Ego vehicle is at the center

Analyze the BEV image to identify:
- Dense clusters indicating vehicles, pedestrians, or objects
- Spatial layout and distances
- Object shapes and orientations
- Scene geometry and structure
- Potential occlusions or gaps in coverage

Be precise and quantitative in your observations."""

        user_prompt = f"""Analyze this LiDAR Bird's Eye View visualization.

Statistical Context:
{json.dumps(bev_description, indent=2)}

Identify all objects, their positions, and spatial relationships."""

        if context:
            user_prompt += f"\n\nAdditional Context from other sensors:\n{json.dumps(context, indent=2)}"
        
        # Build message with image
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
        
        return self.call_llm(messages, temperature=0.3)
    
    def _combine_insights(self, llm_insights: str, vision_insights: str, 
                         bev_description: Dict) -> str:
        """Combine insights from both processing methods"""
        
        system_prompt = """You are a sensor fusion expert combining multiple analyses of LiDAR data.
You have insights from:
1. Statistical analysis of the raw point cloud
2. Visual analysis of the Bird's Eye View representation

Synthesize these into a single, comprehensive description that:
- Resolves any contradictions
- Combines complementary information
- Provides the most complete scene understanding
- Maintains precision and quantitative details

Be concise but comprehensive."""

        user_prompt = f"""Combine these two analyses of the same LiDAR scan:

Statistical Analysis:
{llm_insights}

Visual BEV Analysis:
{vision_insights}

Point Cloud Stats:
{json.dumps(bev_description, indent=2)}

Provide a unified, comprehensive analysis."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_llm(messages, temperature=0.4)
    
    def _generate_bev_visualization(self, point_cloud: np.ndarray, 
                                   resolution: int = 512,
                                   x_range: tuple = (-50, 50),
                                   y_range: tuple = (-50, 50)) -> np.ndarray:
        """
        Generate Bird's Eye View visualization of point cloud
        
        Args:
            point_cloud: Nx4 array (x, y, z, intensity)
            resolution: Image resolution (pixels)
            x_range: (min, max) range in meters for x-axis
            y_range: (min, max) range in meters for y-axis
            
        Returns:
            RGB image as numpy array
        """
        # Create BEV grid
        bev = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Filter points within range
        mask = (
            (point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1]) &
            (point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1])
        )
        filtered_points = point_cloud[mask]
        
        if len(filtered_points) == 0:
            # Return empty image if no points
            return np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # Convert coordinates to pixel indices
        x_pixels = ((filtered_points[:, 0] - x_range[0]) / 
                   (x_range[1] - x_range[0]) * resolution).astype(int)
        y_pixels = ((filtered_points[:, 1] - y_range[0]) / 
                   (y_range[1] - y_range[0]) * resolution).astype(int)
        
        # Clip to valid range
        x_pixels = np.clip(x_pixels, 0, resolution - 1)
        y_pixels = np.clip(y_pixels, 0, resolution - 1)
        
        # Accumulate points (density map)
        for x, y in zip(x_pixels, y_pixels):
            bev[y, x] += 1
        
        # Normalize and convert to RGB with colormap
        bev = np.log1p(bev)  # Log scale for better visualization
        bev = (bev / bev.max() * 255).astype(np.uint8) if bev.max() > 0 else bev.astype(np.uint8)
        
        # Apply colormap (hot colormap: black -> red -> yellow -> white)
        # Create RGB image
        bev_rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        bev_rgb[:, :, 0] = bev  # Red channel
        bev_rgb[:, :, 1] = np.clip(bev.astype(float) * 0.7, 0, 255).astype(np.uint8)  # Green
        bev_rgb[:, :, 2] = np.clip(bev.astype(float) * 0.3, 0, 255).astype(np.uint8)  # Blue
        
        # Add ego vehicle marker (center)
        center = resolution // 2
        marker_size = 10
        bev_rgb[center-marker_size:center+marker_size, 
                center-2:center+2] = [0, 255, 0]  # Green vertical line
        bev_rgb[center-2:center+2, 
                center-marker_size:center+marker_size] = [0, 255, 0]  # Green horizontal line
        
        # Flip vertically (so forward is up)
        bev_rgb = np.flipud(bev_rgb)
        
        return bev_rgb
    
    @staticmethod
    def _analyze_point_cloud(pc: np.ndarray) -> Dict[str, Any]:
        """Extract statistical features from point cloud"""
        
        # Basic stats
        stats = {
            "num_points": len(pc),
            "spatial_extent": {
                "x_range": [float(pc[:, 0].min()), float(pc[:, 0].max())],
                "y_range": [float(pc[:, 1].min()), float(pc[:, 1].max())],
                "z_range": [float(pc[:, 2].min()), float(pc[:, 2].max())]
            },
            "mean_intensity": float(pc[:, 3].mean()) if pc.shape[1] > 3 else None
        }
        
        # Enhanced analysis: point density by region
        # Divide space into regions (front, back, left, right)
        front_mask = pc[:, 1] > 0
        back_mask = pc[:, 1] <= 0
        left_mask = pc[:, 0] < 0
        right_mask = pc[:, 0] >= 0
        
        stats["regional_density"] = {
            "front": int(np.sum(front_mask)),
            "back": int(np.sum(back_mask)),
            "left": int(np.sum(left_mask)),
            "right": int(np.sum(right_mask)),
            "front_left": int(np.sum(front_mask & left_mask)),
            "front_right": int(np.sum(front_mask & right_mask)),
            "back_left": int(np.sum(back_mask & left_mask)),
            "back_right": int(np.sum(back_mask & right_mask))
        }
        
        # Height-based segmentation (rough ground vs obstacles)
        ground_mask = pc[:, 2] < -1.0  # Below ego vehicle
        obstacle_mask = pc[:, 2] > 0.5  # Above ground
        
        stats["height_segmentation"] = {
            "ground_points": int(np.sum(ground_mask)),
            "obstacle_points": int(np.sum(obstacle_mask)),
            "obstacle_ratio": float(np.sum(obstacle_mask) / len(pc))
        }
        
        # Distance-based analysis
        distances = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        stats["distance_distribution"] = {
            "mean_distance": float(distances.mean()),
            "max_distance": float(distances.max()),
            "points_within_10m": int(np.sum(distances < 10)),
            "points_within_30m": int(np.sum(distances < 30)),
            "points_beyond_30m": int(np.sum(distances >= 30))
        }
        
        return stats
    
    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


# TODO: CANT DEPLOY QWEN2-VL VIA AZURE STUDENT SUBSCRIPTION
# class QwenLiDARAgent(LiDARAgent):
#     """
#     Specialized LiDAR agent using Qwen2-VL
#     Qwen2-VL is better at spatial reasoning than GPT-4o
#     """
    
#     def __init__(self, client, model: str, agent_name: str):
#         # Force vision-based processing
#         super().__init__(client, model, agent_name, backend="qwen")
        
#     def call_llm(self, messages, temperature=0.3):
#         """
#         Override to use Qwen2-VL specific parameters if needed
#         Qwen2-VL may have different API parameters than GPT models
#         """
#         # If using Qwen through Azure, adjust parameters as needed
#         # For now, use standard OpenAI API format
#         return super().call_llm(messages, temperature)


# Factory function to create appropriate LiDAR agent
def create_lidar_agent(client, model_config, backend: str = "hybrid"):
    """
    Factory function to create LiDAR agent with appropriate backend
    
    Args:
        client: Azure OpenAI client
        model_config: ModelConfig object
        backend: "llm", "qwen" (not implemented), or "hybrid"
        
    Returns:
        Configured LiDARAgent instance
    """
    # # Choose model based on backend
    # if backend == "qwen":
    #     # Use Qwen2-VL if available
    #     model = "qwen-vl"
    #     return QwenLiDARAgent(client, model, "QwenLiDARAgent")
    
    if backend == "hybrid":
        # Use vision-capable model (GPT-4o or GPT-4o-mini with vision)
        model = model_config.vision_model
        return LiDARAgent(client, model, "HybridLiDARAgent", backend="hybrid")
    
    else:  # backend == "llm"
        # Use cheaper model for statistics only
        model = model_config.small_model
        return LiDARAgent(client, model, "StatisticalLiDARAgent", backend="llm")
