from typing import List, Dict, Any
import json
from agents import BaseAgent

class SuggesterAgent(BaseAgent):
    def suggest(self, features: Dict[str, Any], iteration: int = 1) -> Dict[str, Any]:
        """
        Suggest refinements to features
        
        Args:
            features: Current features (can be seed features or refined features)
            iteration: Current iteration number (affects prompting)
        """
        
        # Adjust prompt based on iteration
        if iteration == 1:
            context = "This is the first review of the initial features."
        else:
            context = f"This is iteration {iteration}. Focus on remaining issues."
        
        system_prompt = f"""You are a quality assurance expert who reviews and suggests improvements.

{context}

Analyze the features and suggest:
- Missing information that should be included
- Redundant or unclear descriptions
- Inconsistencies between different aspects
- Areas needing more detail or precision
- Better ways to structure the information

IMPORTANT: 
- If the features are already high quality and comprehensive, state: "No further suggestions needed."
- Be specific and constructive
- Only suggest meaningful improvements
- Avoid nitpicking minor issues if overall quality is good"""

        user_prompt = f"""Review these features (Iteration {iteration}):

{json.dumps(features, indent=2)}

Provide specific, actionable suggestions for refinement.
If the features are already comprehensive and well-structured, explicitly state that no further improvements are needed."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.6)
        
        return {
            "agent": self.agent_name,
            "iteration": iteration,
            "suggestions": response
        }