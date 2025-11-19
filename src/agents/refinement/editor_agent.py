from typing import List, Dict, Any
import json
from agents import BaseAgent

class EditorAgent(BaseAgent):
    """Refines features based on suggestions"""
    
    def refine(self, seed_features: List[Dict], suggestions: Dict) -> Dict[str, Any]:
        """Refine features based on suggestions"""
        
        system_prompt = """You are an expert editor who refines and improves feature descriptions.
Apply the suggested improvements to create polished, comprehensive features.
Ensure:
- Completeness and accuracy
- Clarity and precision
- Consistency across all aspects
- Proper structure and organization
- Removal of redundancy

Output refined features in a structured format."""

        features_summary = {f["focus_area"]: f["features"] for f in seed_features}
        
        user_prompt = f"""Refine these features based on the suggestions:

Original Features:
{json.dumps(features_summary, indent=2)}

Suggestions:
{suggestions['suggestions']}

Provide refined, high-quality features."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.5)
        
        return {
            "agent": self.agent_name,
            "refined_features": response
        }