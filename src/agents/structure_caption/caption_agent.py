from typing import Any, Dict
import json
from agents import BaseLLMAgent

class CaptionGenerator(BaseLLMAgent):
    """Generates final structured captions"""
    
    def generate_structured_caption(self, refined_features: Dict) -> Dict[str, Any]:
        """Generate structured JSON caption"""
        
        system_prompt = """You are a caption generation expert for autonomous driving scenes.
Generate a structured JSON caption following this schema:

{
  "scene_summary": "Brief overall description",
  "ego_vehicle": {
    "action": "current action (e.g., driving, turning, stopped)",
    "lane_position": "position in lane",
    "speed_estimate": "estimated speed category"
  },
  "objects": [
    {
      "category": "object type",
      "position": "relative position (front/back/left/right, distance)",
      "state": "static/moving/stopped",
      "attributes": ["relevant attributes"],
      "visibility": "visibility level"
    }
  ],
  "road_structure": {
    "type": "intersection/straight/curve/etc",
    "lanes": "number and configuration",
    "markings": ["visible markings"]
  },
  "environment": {
    "lighting": "day/night/dusk/dawn",
    "weather": "clear/rain/etc",
    "location_type": "urban/highway/residential"
  },
  "safety_critical": ["list of safety-relevant observations"]
}

Be precise and comprehensive."""

        user_prompt = f"""Generate a structured caption from these refined features:

{refined_features['refined_features']}

Output valid JSON only, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.3)
        
        # Try to parse as JSON
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            caption_json = json.loads(cleaned)
        except json.JSONDecodeError:
            caption_json = {"raw_output": response, "parse_error": True}
        
        return {
            "agent": self.agent_name,
            "structured_caption": caption_json,
            "raw_caption": response
        }
    
    def answer_mqa_question(self, question: str, structured_caption: Dict, 
                           refined_features: Dict) -> str:
        """Answer nuScenes-MQA style question"""
        
        system_prompt = """You are an expert at answering questions about driving scenes.
Answer using the structured caption and features available.
Follow the nuScenes-MQA format:
- Use XML tags:
  - <target>: Encapsulates <cnt> and <obj>.
  - <obj>: Represents an object, restricted to a single word.
  - <cnt>: Represents a count, restricted to a single word.
  - <ans>: Represents a binary response, a single word.
  - <cam>: Represents one of the six cameras.
  - <dst>: Represents distance.
  - <loc>: Represents (x, y) coordinates.
- Be precise with counts and object references
- Use the exact format expected by the benchmark

Examples:
Q: "How many <obj>cars</obj> are in <cam>front</cam>?"
A: "There are <target><cnt>2</cnt> <obj>cars</obj></target>."
"""

        user_prompt = f"""Question: {question}

Scene Information:
{json.dumps(structured_caption, indent=2)}

Additional Features:
{refined_features['refined_features'][:500]}

Provide a precise answer in the correct format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_llm(messages, temperature=0.2)