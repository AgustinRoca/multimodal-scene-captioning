from typing import List, Dict, Any
import json
from agents import BaseAgent
from pydantic import BaseModel, Field
from agents.refinement.suggester_agent import SuggestionResponse

class RefinedFeaturesResponse(BaseModel):
    """Structured response from Editor agent"""
    refined_features: Dict[str, str] = Field(
        description="Dictionary of refined features by focus area"
    )
    changes_made: List[str] = Field(
        default_factory=list,
        description="List of key changes applied"
    )

class EditorAgent(BaseAgent):
    """Enhanced editor that returns structured JSON"""
    
    def refine(self, features: Dict[str, Any], 
               suggestion_response: SuggestionResponse, 
               iteration: int = 1) -> RefinedFeaturesResponse:
        """
        Refine features based on suggestions with structured output
        
        Args:
            features: Current features
            suggestion_response: SuggestionResponse from SuggesterAgent
            iteration: Current iteration number
            
        Returns:
            RefinedFeaturesResponse with structured refined features
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

You must respond with valid JSON matching this schema:
{{
  "refined_features": {{
    "focus_area_1": "refined description",
    "focus_area_2": "refined description",
    ...
  }},
  "changes_made": ["change 1", "change 2", ...]
}}"""

        suggestions_text = "\n".join(f"- {s}" for s in suggestion_response.suggestions)
        
        user_prompt = f"""Refine these features based on the suggestions:

Current Features:
{json.dumps(features, indent=2)}

Suggestions:
{suggestions_text}

Reasoning: {suggestion_response.reasoning}

Provide refined features in JSON format with the same structure (same focus area keys)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.5)
        
        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return RefinedFeaturesResponse(**data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Failed to parse editor response as JSON: {e}")
            print(f"  Raw response: {response[:200]}...")
            # Return current features unchanged if parsing fails
            return RefinedFeaturesResponse(
                refined_features=features,
                changes_made=["Failed to parse response, no changes applied"]
            )
