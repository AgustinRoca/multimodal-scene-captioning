from typing import Any, Dict, List, Optional
from agents import BaseAgent
import numpy as np
import json
from PIL import Image
from io import BytesIO
import base64

class CameraAgent(BaseAgent):
    """Processes camera images to extract visual features"""
    
    def process(self, images: List[np.ndarray], camera_names: List[str], 
                context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process all camera views in a single call.
        """
        system_prompt = """You are a camera vision expert analyzing driving scenes from multiple camera views.
You will be provided with multiple camera images from different viewpoints around a vehicle.

For EACH camera view, describe what you see focusing on:
- Vehicles (type, position, movement)
- Pedestrians and cyclists
- Road structure and markings
- Traffic signs and signals
- Environmental conditions
- Potential hazards

Be precise and structured in your observations for each camera.

CRITICAL INSTRUCTIONS:
- Analyze EACH camera view separately and thoroughly
- Include EVERY piece of information available - no summarization
- Be exhaustive and thorough - longer captions with more detail are better
- Don't say "various objects" or "several vehicles" - name each one specifically
- Include all numerical data (distances, counts, positions)
- Write as if you're describing the scene to someone who can't see it
- Format your response with clear camera view labels"""

        # Build user content with all images
        user_content = []
        
        # Add context if available
        if context:
            user_content.append({
                "type": "text",
                "text": f"Context from other sensors:\n{json.dumps(context, indent=2)}\n\n"
            })
        
        user_content.append({
            "type": "text",
            "text": f"Analyze all {len(camera_names)} camera views. For each view, provide detailed observations:\n\n"
        })
        
        # Add all camera images
        for img, name in zip(images, camera_names):
            b64_img = self._image_to_base64(img)
            user_content.append({
                "type": "text",
                "text": f"Camera: {name}"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "low"  # Use low detail to reduce tokens
                }
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Single LLM call with all cameras
        response = self.call_llm(messages, temperature=0.3)
        
        # Parse response into individual camera observations
        # Try to split by camera names, fallback to returning as single observation
        observations = {}
        response_lower = response.lower()
        
        # Try to identify sections for each camera
        found_sections = False
        for name in camera_names:
            name_lower = name.lower()
            if name_lower in response_lower:
                found_sections = True
                break
        
        if found_sections:
            # Parse sections for each camera
            for i, name in enumerate(camera_names):
                # Find the section for this camera
                name_patterns = [name.lower(), name.replace('_', ' ').lower()]
                section_start = -1
                
                for pattern in name_patterns:
                    idx = response_lower.find(pattern)
                    if idx != -1:
                        section_start = idx
                        break
                
                if section_start != -1:
                    # Find where the next camera section starts
                    section_end = len(response)
                    for next_name in camera_names[i+1:]:
                        next_patterns = [next_name.lower(), next_name.replace('_', ' ').lower()]
                        for pattern in next_patterns:
                            idx = response_lower.find(pattern, section_start + 1)
                            if idx != -1 and idx < section_end:
                                section_end = idx
                                break
                    
                    observations[name] = response[section_start:section_end].strip()
                else:
                    observations[name] = f"(Analysis for {name} not clearly separated in response)"
        else:
            # Couldn't parse sections, return full response for each camera
            observations = {name: response for name in camera_names}

        return {
            "agent": self.agent_name,
            "modality": "camera",
            "camera_views": camera_names,
            "observations": observations,
            "full_response": response  # Keep full response for reference
        }

    
    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()