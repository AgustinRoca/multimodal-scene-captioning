from typing import List, Dict, Any
import json
from agents import BaseAgent

class SuggesterAgent(BaseAgent):
    """Suggests improvements to seed features"""
    
    def suggest(self, seed_features: List[Dict]) -> Dict[str, Any]:
        """Suggest refinements to seed features"""
        
        system_prompt = """You are a quality assurance expert who reviews and suggests improvements.
Analyze the extracted features and suggest:
- Missing information that should be included
- Redundant or unclear descriptions
- Inconsistencies between different aspects
- Areas needing more detail or precision
- Better ways to structure the information

Be specific and constructive."""

        features_summary = {f["focus_area"]: f["features"] for f in seed_features}
        
        user_prompt = f"""Review these extracted features and suggest improvements:

{json.dumps(features_summary, indent=2)}

Provide specific, actionable suggestions for refinement."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.7)
        
        return {
            "agent": self.agent_name,
            "suggestions": response
        }