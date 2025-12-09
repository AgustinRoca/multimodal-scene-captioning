from typing import List, Dict, Any
import json
from agents import BaseAgent
from pydantic import BaseModel, Field

class SuggestionResponse(BaseModel):
    """Structured response from Suggester agent"""
    has_suggestions: bool = Field(
        description="True if there are meaningful suggestions, False if features are complete"
    )
    suggestions: List[str] = Field(
        description="List of specific improvement suggestions"
    )
    reasoning: str = Field(
        description="Brief explanation of the suggestions or why no suggestions are needed"
    )

class SuggesterAgent(BaseAgent):
    """Enhanced suggester that returns structured JSON"""
    
    def suggest(self, features: Dict[str, Any], iteration: int = 1) -> SuggestionResponse:
        """
        Suggest refinements to features with structured output
        
        Args:
            features: Current features (can be seed features or refined features)
            iteration: Current iteration number (affects prompting)
            
        Returns:
            SuggestionResponse with structured suggestions
        """
        
        # Adjust prompt based on iteration
        if iteration == 1:
            context = "This is the first review of the initial features."
        else:
            context = f"This is iteration {iteration}. Focus on remaining issues only."
        
        system_prompt = f"""You are a quality assurance expert who reviews and suggests improvements.

{context}

Analyze the features and suggest improvements focusing on:
- Missing information that should be included
- Redundant or unclear descriptions
- Inconsistencies between different aspects
- Areas needing more detail or precision
- Better ways to structure the information

IMPORTANT: 
- If the features are already high quality and comprehensive, set has_suggestions to false
- Be specific and constructive
- Only suggest meaningful improvements
- Avoid nitpicking minor issues if overall quality is good

You must respond with valid JSON matching this schema:
{{
  "has_suggestions": boolean,
  "suggestions": ["suggestion 1", "suggestion 2", ...],
  "reasoning": "brief explanation"
}}"""

        user_prompt = f"""Review these features (Iteration {iteration}):

{json.dumps(features, indent=2)}

Analyze and provide structured suggestions in JSON format.
If features are comprehensive, set has_suggestions to false and explain why."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.6, response_format=SuggestionResponse)
        
        return response
