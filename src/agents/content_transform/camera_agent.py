from typing import Any, Dict, List, Optional
from agents import BaseLLMAgent
import numpy as np
import json
from PIL import Image
from io import BytesIO
import base64

class CameraAgent(BaseLLMAgent):
    """Processes camera images to extract visual features"""
    
    def process(self, images: List[np.ndarray], camera_names: List[str], 
                context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process each camera view independently to avoid rate limits.
        """
        observations = {}

        system_prompt = """You are a camera vision expert analyzing driving scenes.
Describe what you see in the camera focusing on:
- Vehicles (type, position, movement)
- Pedestrians and cyclists
- Road structure and markings
- Traffic signs and signals
- Environmental conditions
- Potential hazards

Be precise and structured in your observations."""

        for img, name in zip(images, camera_names):

            # Convert image to base64
            b64_img = self._image_to_base64(img)

            user_prompt = f"Analyze the camera view: {name}."
            
            if context:
                user_prompt += f"\n\nContext from other sensors:\n{json.dumps(context, indent=2)}"

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            }
                        }
                    ]
                }
            ]

            # One LLM call per camera (Rate limit reached if sending all at once)
            observations[name] = self.call_llm(messages, temperature=0.3)

        return {
            "agent": self.agent_name,
            "modality": "camera",
            "camera_views": camera_names,
            "observations": observations
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