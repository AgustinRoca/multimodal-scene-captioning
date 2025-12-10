from typing import Any, Dict, List
import json
from agents.base_agent import BaseAgent
from openai import AzureOpenAI


class FocusedCaptionAgent(BaseAgent):
    """Generates detailed captions from transformed content for specific focus area"""
    
    def __init__(self, client: AzureOpenAI, model: str, agent_name: str, focus_area: str):
        super().__init__(client, model, agent_name)
        self.focus_area = focus_area
    
    def generate(self, transformed_content: Dict) -> Dict[str, Any]:
        """Generate comprehensive caption focused on specific aspect"""
        
        focus_prompts = {
            "objects": """Write a comprehensive description of ALL objects in the scene. 
Include every single object detected - vehicles (specify types: cars, trucks, buses, trailers), 
pedestrians (adults, children, their locations), cyclists, motorcycles, barriers, traffic cones, 
construction equipment, and any other objects. For each object mention: its type, approximate 
position/direction from ego vehicle, distance if known, state (moving/stopped/parked), and any 
notable attributes. Don't summarize - describe each object individually with all available details.""",
            
            "scene_structure": """Write a comprehensive description of the road structure and environment. 
Include: road type (highway, urban street, intersection, etc.), number of lanes, lane markings, 
ego vehicle's lane position, road surface condition, presence of sidewalks, crosswalks, intersections, 
traffic lights, road signs, road geometry (straight, curved, uphill, downhill), visible infrastructure 
(guardrails, barriers, street lights), parking areas, and any other structural elements. Be exhaustive 
and mention every visible element.""",
            
            "spatial_relations": """Write a comprehensive description of spatial relationships in the scene. 
Describe the position of every object relative to the ego vehicle (front/back/left/right and approximate 
distances in meters). Describe objects' positions relative to each other. Mention which objects are close 
together, which are far apart. Describe the spatial layout - which lane each vehicle is in, relative 
positions of pedestrians to vehicles, clustering of objects. Include all distance information and 
directional relationships. Be thorough and don't omit any spatial detail.""",
            
            "dynamics": """Write a comprehensive description of all motion and dynamic aspects in the scene. 
Describe which objects are moving and which are stationary. For moving objects, describe their direction 
of movement, approximate speed (slow, moderate, fast), trajectory, and any changes in motion. Mention if 
vehicles are accelerating, braking, turning. Describe pedestrian movement patterns. Note any stopped 
vehicles and whether they appear to be parked or temporarily stopped. Include temporal aspects and 
motion predictions if relevant. Describe every dynamic element comprehensively.""",
            
            "safety": """Write a comprehensive description of safety-critical elements and potential risks. 
Identify all objects that could pose safety concerns: close vehicles (especially in blind spots), 
pedestrians near or crossing the road, cyclists in traffic, vehicles changing lanes, objects in the 
ego vehicle's path, stopped vehicles, construction zones, poor visibility areas, vulnerable road users, 
unexpected obstacles, and any unusual or hazardous situations. For each safety concern, explain why it's 
critical and what attention it requires. Be thorough and mention every potential safety consideration."""
        }
        
        system_prompt = f"""You are an expert at writing comprehensive, detailed captions for autonomous driving scenes.
Your focus area is: {self.focus_area}

CRITICAL INSTRUCTIONS:
- Write in complete, flowing sentences using natural language
- Include EVERY piece of information available - no summarization
- Don't use bullet points, lists, or structured formatting - write in prose
- Be exhaustive and thorough - longer captions with more detail are better
- Don't say "various objects" or "several vehicles" - name each one specifically
- Include all numerical data (distances, counts, positions)
- Write as if you're describing the scene to someone who can't see it

{focus_prompts.get(self.focus_area, '')}"""

        # Prepare content - extract relevant information
        observations = transformed_content.get('observations', [])
        observations_text = '\n\n'.join([str(obs) for obs in observations if obs])
        
        user_prompt = f"""Write a comprehensive {self.focus_area} caption for this autonomous driving scene.

Scene Information:
{observations_text}

Write a detailed, flowing caption that includes every single detail about {self.focus_area}. 
Don't omit anything - include all objects, all positions, all states, all relationships. 
Write in natural prose, not lists. Be as thorough and complete as possible."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.6)
        
        return {
            "agent": self.agent_name,
            "focus_area": self.focus_area,
            "caption": response
        }


class ComprehensiveCaptionMerger(BaseAgent):
    """Merges all focused captions into a single comprehensive scene caption"""
    
    def merge_captions(self, focused_captions: List[Dict[str, Any]]) -> str:
        """
        Merge multiple focused captions into one comprehensive caption
        
        Args:
            focused_captions: List of caption dicts with 'focus_area' and 'caption'
            
        Returns:
            Single comprehensive caption string
        """
        
        system_prompt = """You are an expert at synthesizing comprehensive scene descriptions for autonomous driving.

Your task is to merge multiple detailed captions (each focusing on a different aspect) into ONE 
single, comprehensive, flowing narrative description of the entire scene.

CRITICAL INSTRUCTIONS:
- Combine ALL information from ALL captions into one unified description
- Write in natural, flowing prose - tell a complete story of the scene
- DO NOT lose any information during the merge - every detail must be preserved
- Eliminate redundancy, but keep all unique information
- Organize the information logically (e.g., describe environment, then static elements, then objects, then dynamics)
- Use transitional phrases to create a cohesive narrative
- The final caption should read as one continuous description, not separate sections
- Longer is better - comprehensiveness is more important than brevity
- Don't use section headers or bullet points - write in flowing paragraphs

The result should read like a detailed scene description written by a single observer, not a 
combination of separate reports."""

        # Prepare all captions
        captions_by_focus = {cap['focus_area']: cap['caption'] for cap in focused_captions}
        
        captions_text = ""
        for focus_area in ['scene_structure', 'objects', 'spatial_relations', 'dynamics', 'safety']:
            if focus_area in captions_by_focus:
                captions_text += f"\n\n{focus_area.upper()} CAPTION:\n{captions_by_focus[focus_area]}"
        
        user_prompt = f"""Merge these detailed captions into ONE comprehensive scene description:
{captions_text}

Create a single, flowing narrative that includes ALL information from all captions. 
Organize it logically but write it as continuous prose. Don't lose any details.
Make it read like one cohesive description of the entire scene."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_llm(messages, temperature=0.5)


class SeedFeatureAgent(BaseAgent):
    """
    Complete system for generating comprehensive scene captions
    
    Uses multiple focused agents then merges into single caption
    """
    
    def __init__(self, client: AzureOpenAI, model: str):
        self.client = client
        self.model = model
        
        # Create focused caption agents
        self.focused_agents = [
            FocusedCaptionAgent(client, model, f"FocusedCaption_{area}", area)
            for area in ["scene_structure", "objects", "spatial_relations", "dynamics", "safety"]
        ]
        
        # Create merger agent
        self.merger = ComprehensiveCaptionMerger(client, model, "CaptionMerger")
    
    def generate_comprehensive_caption(self, transformed_content: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive caption from transformed content
        
        Args:
            transformed_content: Output from Layer 1 agents
            
        Returns:
            Dict with final comprehensive caption and intermediate captions
        """
                
        # Step 1: Generate focused captions
        focused_captions = []
        for agent in self.focused_agents:
            caption_output = agent.generate(transformed_content)
            focused_captions.append(caption_output)
            print(f"  ✓ {agent.agent_name} generated {agent.focus_area} caption "
                  f"({len(caption_output['caption'])} chars)")
        
        # Step 2: Merge into single comprehensive caption
        print("  Merging all captions into comprehensive description...")
        final_caption = self.merger.merge_captions(focused_captions)
        print(f"  ✓ Final comprehensive caption generated ({len(final_caption)} chars)")
        
        return {
            "focused_captions": focused_captions,
            "final_caption": final_caption
        }


# Factory function
def create_seed_feature_agent(client: AzureOpenAI, model: str):
    """Create comprehensive caption generation system"""
    return SeedFeatureAgent(client, model)