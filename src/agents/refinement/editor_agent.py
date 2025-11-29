from typing import List, Dict, Any
import json
from agents import BaseAgent

class EditorAgent(BaseAgent):
    """Enhanced editor that tracks refinement quality"""
    
    def refine(self, features: Dict[str, Any], suggestions: Dict[str, Any], 
               iteration: int = 1) -> Dict[str, Any]:
        """
        Refine features based on suggestions
        
        Args:
            features: Current features
            suggestions: Suggestions dict from SuggesterAgent
            iteration: Current iteration number
        """
        
        system_prompt = f"""You are an expert editor who refines and improves feature descriptions.

This is refinement iteration {iteration}.

Apply the suggested improvements to create polished, comprehensive features.
Ensure:
- Completeness and accuracy
- Clarity and precision
- Consistency across all aspects
- Proper structure and organization
- Removal of redundancy

Output refined features in a structured format matching the input structure."""

        user_prompt = f"""Refine these features based on the suggestions:

Current Features:
{json.dumps(features, indent=2)}

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
            "iteration": iteration,
            "refined_features": response
        }