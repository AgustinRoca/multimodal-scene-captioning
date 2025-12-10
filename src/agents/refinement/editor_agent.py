from typing import List, Dict, Any, Optional
import json
from agents import BaseAgent
from pydantic import BaseModel, Field
from agents.refinement.suggester_agent import SuggestionResponse


class RefinedFeaturesResponse(BaseModel):
    """Structured response from Editor agent"""
    caption: str = Field(description="The refined caption text")
    changes_made: List[str] = Field(description="List of key changes applied")


class EditorAgent(BaseAgent):
    """Enhanced editor that returns structured JSON"""
    
    def refine(self, caption: str, 
               suggestion_response: SuggestionResponse,
               transformed_content: Dict[str, Any],
               iteration: int = 1) -> Dict[str, Any]:
        """
        Refine features based on suggestions with structured output
        
        Args:
            caption: Current caption
            suggestion_response: SuggestionResponse from SuggesterAgent
            iteration: Current iteration number
            
        Returns:
            Dictionary with refined features
        """
        
        system_prompt = f"""You are an expert editor who refines and improves feature descriptions.

This is refinement iteration {iteration}.

Apply the suggested improvements to create polished, comprehensive captions.
Ensure:
- Completeness and accuracy
- Clarity and precision
- Consistency across all aspects
- Proper structure and organization
- Removal of redundancy

CRITICAL INSTRUCTIONS:
- Include EVERY piece of information available - no summarization
- Be exhaustive and thorough - longer captions with more detail are better
- Don't say "various objects" or "several vehicles" - name each one specifically
- Include all numerical data (distances, counts, positions)
- Write as if you're describing the scene to someone who can't see it"""

        suggestions_text = "\n".join(f"- {s}" for s in suggestion_response.suggestions)
        
        user_prompt = f"""Refine this caption based on the suggestions.:

Current Caption:
{caption}

Suggestions:
{suggestions_text}

Reasoning: {suggestion_response.reasoning}

To refine the caption, consider the full context from all sensors:
{json.dumps(transformed_content, indent=2)}

Provide a refined caption and list all changes made."""

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
                "refined_caption": response.caption,
                "changes_made": response.changes_made
            }
            
        except Exception as e:
            print(f"  ⚠️  Error in EditorAgent, returning unchanged features: {e}")
            # Fallback: return features unchanged
            return {
                "refined_caption": caption,
                "changes_made": ["Error occurred, no changes applied"]
            }