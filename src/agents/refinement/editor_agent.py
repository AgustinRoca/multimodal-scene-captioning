from typing import List, Dict, Any, Optional
import json
from agents import BaseAgent
from pydantic import BaseModel, Field
from agents.refinement.suggester_agent import SuggestionResponse


class RefinedFeaturesResponse(BaseModel):
    """Structured response from Editor agent"""
    objects: str = Field(description="Refined description of objects in the scene")
    scene_structure: str = Field(description="Refined description of scene structure")
    spatial_relations: str = Field(description="Refined description of spatial relationships")
    dynamics: str = Field(description="Refined description of dynamics and movement")
    safety: str = Field(description="Refined description of safety considerations")
    changes_made: List[str] = Field(description="List of key changes applied")


class EditorAgent(BaseAgent):
    """Enhanced editor that returns structured JSON"""
    
    def refine(self, features: Dict[str, Any], 
               suggestion_response: SuggestionResponse, 
               iteration: int = 1) -> Dict[str, Any]:
        """
        Refine features based on suggestions with structured output
        
        Args:
            features: Current features
            suggestion_response: SuggestionResponse from SuggesterAgent
            iteration: Current iteration number
            
        Returns:
            Dictionary with refined features
        """
        
        system_prompt = f"""You are an expert editor who refines and improves feature descriptions.

This is refinement iteration {iteration}.

Apply the suggested improvements to create polished, comprehensive features.
Ensure:
- Completeness and accuracy
- Clarity and precision
- Consistency across all aspects
- Proper structure and organization
- Removal of redundancy"""

        suggestions_text = "\n".join(f"- {s}" for s in suggestion_response.suggestions)
        
        user_prompt = f"""Refine these features based on the suggestions:

Current Features:
{json.dumps(features, indent=2)}

Suggestions:
{suggestions_text}

Reasoning: {suggestion_response.reasoning}

Provide refined features for all five focus areas and list all changes made."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_llm(
                messages, 
                temperature=0.5, 
                response_format=RefinedFeaturesResponse
            )
            
            # Convert back to dict format
            return {
                "refined_features": {
                    "objects": response.objects,
                    "scene_structure": response.scene_structure,
                    "spatial_relations": response.spatial_relations,
                    "dynamics": response.dynamics,
                    "safety": response.safety
                },
                "changes_made": response.changes_made
            }
            
        except Exception as e:
            print(f"  ⚠️  Error in EditorAgent, returning unchanged features: {e}")
            # Fallback: return features unchanged
            return {
                "refined_features": features,
                "changes_made": ["Error occurred, no changes applied"]
            }