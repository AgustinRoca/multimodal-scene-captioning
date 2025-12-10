from typing import Any, Dict, List
import json
from agents import BaseAgent
from pydantic import BaseModel, Field


class EgoVehicle(BaseModel):
    """Ego vehicle state"""
    action: str = Field(description="Current action (e.g., driving, turning, stopped)")
    lane_position: str = Field(description="Position in lane (center, left, right)")
    speed_estimate: str = Field(description="Estimated speed category (slow, moderate, fast)")


class SceneObject(BaseModel):
    """Detected object in the scene"""
    category: str = Field(description="Object type (car, truck, pedestrian, etc.)")
    position: str = Field(description="Relative position (front/back/left/right, distance)")
    state: str = Field(description="Object state (static, moving, stopped)")
    attributes: List[str] = Field(description="Relevant attributes")
    visibility: str = Field(description="Visibility level (high, medium, low)")


class RoadStructure(BaseModel):
    """Road structure information"""
    type: str = Field(description="Road type (intersection, straight, curve, etc.)")
    lanes: str = Field(description="Number and configuration of lanes")
    markings: List[str] = Field(description="Visible road markings")


class Environment(BaseModel):
    """Environmental conditions"""
    lighting: str = Field(description="Lighting conditions (day, night, dusk, dawn)")
    weather: str = Field(description="Weather conditions (clear, rain, fog, etc.)")
    location_type: str = Field(description="Location type (urban, highway, residential)")


class StructuredCaption(BaseModel):
    """Complete structured caption for autonomous driving scene"""
    scene_summary: str = Field(description="Brief overall description of the scene")
    ego_vehicle: EgoVehicle = Field(description="Ego vehicle state and action")
    objects: List[SceneObject] = Field(description="List of detected objects in the scene")
    road_structure: RoadStructure = Field(description="Road structure and layout")
    environment: Environment = Field(description="Environmental conditions")
    safety_critical: List[str] = Field(description="List of safety-relevant observations")


class CaptionGenerator(BaseAgent):
    """Generates final structured captions"""
    
    def generate_structured_caption(self, refined_caption: str) -> Dict[str, Any]:
        """Generate structured JSON caption using Pydantic"""
        
        system_prompt = """You are a caption generation expert for autonomous driving scenes.

Generate a comprehensive structured caption based on the refined features provided.

Guidelines:
- scene_summary: Provide a concise 1-2 sentence overview
- ego_vehicle: Describe the ego vehicle's current action, lane position, and estimated speed
- objects: List ALL detected objects with their categories, positions, states, attributes, and visibility
- road_structure: Describe the road type, number of lanes, and visible markings
- environment: Specify lighting, weather, and location type
- safety_critical: List any safety-relevant observations (close objects, hazards, etc.)

Be precise, comprehensive, and factual based on the features provided."""

        user_prompt = f"""Generate a structured caption from this refined caption:

{refined_caption}

Create a complete, accurate caption covering all aspects of the scene."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_llm(
                messages, 
                temperature=0.3, 
                response_format=StructuredCaption
            )
            
            # Convert Pydantic model to dict
            caption_dict = response.model_dump()
            caption_dict["full_caption"] = refined_caption  # Include full caption text
            
            return {
                "agent": self.agent_name,
                "structured_caption": caption_dict
            }
            
        except Exception as e:
            print(f"  ⚠️  Error generating structured caption: {e}")
            # Return minimal fallback caption
            return {
                "agent": self.agent_name,
                "structured_caption": {
                    "scene_summary": "Error generating caption",
                    "full_caption": "Error generating caption",
                    "ego_vehicle": {
                        "action": "unknown",
                        "lane_position": "unknown",
                        "speed_estimate": "unknown"
                    },
                    "objects": [],
                    "road_structure": {
                        "type": "unknown",
                        "lanes": "unknown",
                        "markings": []
                    },
                    "environment": {
                        "lighting": "unknown",
                        "weather": "unknown",
                        "location_type": "unknown"
                    },
                    "safety_critical": ["Caption generation failed"]
                },
                "parse_error": True,
                "error_message": str(e)
            }
    
    def answer_mqa_question(self, question: str, structured_caption: Dict) -> str:
        """Answer nuScenes-MQA style question"""
        
        system_prompt = """You are an expert at answering questions about driving scenes.

Answer using the structured caption and features available.

Follow the nuScenes-MQA format strictly:
- Use XML tags:
  - <target>: Encapsulates <cnt> and <obj>
  - <obj>: Object name (single word or short phrase)
  - <cnt>: Count (number)
  - <ans>: Binary response (yes/no)
  - <cam>: Camera name (front, back, front left, etc.)
  - <dst>: Distance description
  - <loc>: Location coordinates (x, y)

Examples:
Q: "How many <obj>cars</obj> are in <cam>front</cam>?"
A: "There are <target><cnt>2</cnt> <obj>cars</obj></target>."

Q: "Is there a <obj>pedestrian</obj> in <cam>front left</cam>?"
A: "<ans>yes</ans>, there is <target><cnt>1</cnt> <obj>pedestrian</obj></target>."

Be precise with counts and use the exact XML format."""
        
        user_prompt = f"""Question: {question}

Scene Information:
{json.dumps(structured_caption, indent=2)}

Provide a precise answer using the correct XML format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_llm(messages, temperature=0.2)
            return response
        except Exception as e:
            print(f"  ⚠️  Error answering MQA question: {e}")
            return "Error: Unable to answer question"