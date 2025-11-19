from typing import Any, Dict, List, Optional
from agents import BaseAgent
import json

class SceneGraphAgent(BaseAgent):
    """Converts object annotations to scene graph representation"""
    
    def process(self, annotations: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process object annotations into scene graph
        
        Args:
            annotations: List of nuScenes object annotations
            context: Optional context from other agents
        """
        # Convert annotations to structured scene graph
        scene_graph = self._build_scene_graph(annotations)
        
        system_prompt = """You are a scene understanding expert who analyzes object annotations.
Interpret the structured object data focusing on:
- Object categories and attributes
- Spatial relationships between objects
- Object states (moving, stopped, parked)
- Visibility and occlusion patterns
- Scene semantics and interactions

Provide high-level scene understanding."""

        user_prompt = f"""Analyze this scene graph with {len(annotations)} detected objects:

{json.dumps(scene_graph, indent=2)}

Provide semantic interpretation of the scene structure and object relationships."""

        if context:
            user_prompt += f"\n\nContext from other sensors:\n{json.dumps(context, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.3)
        
        return {
            "agent": self.agent_name,
            "modality": "scene_graph",
            "scene_graph": scene_graph,
            "observations": response
        }
    
    @staticmethod
    def _build_scene_graph(annotations: List[Dict]) -> Dict[str, Any]:
        """Build structured scene graph from annotations"""
        # Group objects by category
        objects_by_category = {}
        for ann in annotations:
            category = ann.get("category_name", "unknown")
            if category not in objects_by_category:
                objects_by_category[category] = []
            
            obj_info = {
                "position": ann.get("translation", [0, 0, 0]),
                "size": ann.get("size", [0, 0, 0]),
                "attributes": ann.get("attribute_tokens", []),
                "visibility": ann.get("visibility_token", "unknown"),
                "num_lidar_pts": ann.get("num_lidar_pts", 0)
            }
            objects_by_category[category].append(obj_info)
        
        return {
            "object_counts": {cat: len(objs) for cat, objs in objects_by_category.items()},
            "objects_by_category": objects_by_category,
            "total_objects": len(annotations)
        }