from typing import Any, Dict
import json
from agents.base_llm_agent import BaseLLMAgent
from openai import AzureOpenAI

class SeedFeatureAgent(BaseLLMAgent):
    """Generates seed features from transformed content"""
    
    def __init__(self, client: AzureOpenAI, model: str, agent_name: str, focus_area: str):
        super().__init__(client, model, agent_name)
        self.focus_area = focus_area
    
    def generate(self, transformed_content: Dict) -> Dict[str, Any]:
        """Generate seed features focused on specific aspect"""
        
        focus_prompts = {
            "objects": "Focus on identifying and describing all objects: vehicles, pedestrians, cyclists, etc.",
            "scene_structure": "Focus on road structure, lanes, intersections, traffic infrastructure.",
            "spatial_relations": "Focus on spatial relationships, distances, and relative positions.",
            "dynamics": "Focus on motion, actions, and dynamic aspects of the scene.",
            "safety": "Focus on safety-critical elements, hazards, and risk assessment."
        }
        
        system_prompt = f"""You are a feature extraction expert specializing in {self.focus_area}.
Extract structured features from the scene observations.
{focus_prompts.get(self.focus_area, '')}

Output your features in a structured format."""

        user_prompt = f"""Extract {self.focus_area} features from this scene:

{json.dumps(transformed_content, indent=2)}

Provide structured, detailed features."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.6)
        
        return {
            "agent": self.agent_name,
            "focus_area": self.focus_area,
            "features": response
        }