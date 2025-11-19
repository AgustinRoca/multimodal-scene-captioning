from typing import Any, Dict, List, Optional
from agents import BaseAgent
import json

class BEVFusionAgent(BaseAgent):
    """Fuses multi-modal information into unified representation"""
    
    def process(self, camera_output: Dict, lidar_output: Dict, 
                scene_graph_output: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fuse information from multiple modalities
        
        Args:
            camera_output: Output from CameraAgent
            lidar_output: Output from LiDARAgent
            scene_graph_output: Optional output from SceneGraphAgent
        """
        system_prompt = """You are a sensor fusion expert who integrates multi-modal information.
Synthesize information from different sensors to create a unified understanding:
- Resolve conflicts between sensor observations
- Combine complementary information
- Identify objects detected by multiple sensors
- Build comprehensive spatial understanding
- Assess confidence based on multi-sensor agreement

Provide a fused, coherent scene representation."""

        inputs = {
            "camera_observations": camera_output.get("observations", ""),
            "lidar_observations": lidar_output.get("observations", "")
        }
        
        if scene_graph_output:
            inputs["scene_graph_observations"] = scene_graph_output.get("observations", "")
        
        user_prompt = f"""Fuse these multi-modal sensor observations into a unified scene understanding:

{json.dumps(inputs, indent=2)}

Create a comprehensive, coherent description that leverages all available information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.5)
        
        return {
            "agent": self.agent_name,
            "modality": "fused",
            "observations": response
        }