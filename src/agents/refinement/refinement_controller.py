"""
Iterative Refinement System for Feature Enhancement
Implements suggest→edit loop until convergence or max iterations
"""

from typing import List, Dict, Any
import json
from dataclasses import dataclass


from agents import SuggesterAgent, EditorAgent

@dataclass
class RefinementIteration:
    """Records one iteration of refinement"""
    iteration: int
    suggestions: List[str]
    has_suggestions: bool
    reasoning: str
    refined_caption: Dict[str, str]
    changes_made: List[str]

class IterativeRefinementController:
    """
    Controls the iterative refinement process
    
    Manages the suggest→edit loop until convergence or max iterations
    """
    
    def __init__(self, suggester: SuggesterAgent, 
                 editor: EditorAgent,
                 max_iterations: int = 5,
                 verbose: bool = True):
        """
        Initialize refinement controller
        
        Args:
            suggester: SuggesterAgent instance
            editor: EditorAgent instance
            max_iterations: Maximum number of refinement iterations
            verbose: Print progress messages
        """
        self.suggester = suggester
        self.editor = editor
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Track refinement history
        self.iterations: List[RefinementIteration] = []
    
    def refine(self, seed_caption: Dict, transformed_content: Dict) -> Dict[str, Any]:
        """
        Run iterative refinement process
        
        Args:
            seed_features: Initial features from seed generation
            
        Returns:
            Dictionary containing:
            - final_features: Best refined features
            - iterations: List of all refinement iterations
            - converged: Whether process converged naturally
            - total_iterations: Number of iterations performed
        """
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("STARTING ITERATIVE REFINEMENT")
            print(f"Max iterations: {self.max_iterations}")
            print(f"{'='*80}\n")

        self.iterations = []
        
        # Convert seed features to dict format for processing
        current_caption = seed_caption
        converged = False
        
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Step 1: Get suggestions
            if self.verbose:
                print("  Suggester: Analyzing features...")
            
            suggestion_response = self.suggester.suggest(current_caption, iteration)
            
            if self.verbose:
                print(f"  Suggester: {len(suggestion_response.suggestions)} suggestions")
                if not suggestion_response.has_suggestions:
                    print(f"  Suggester: ✓ No more suggestions - {suggestion_response.reasoning}")
            
            # Record iteration (will update refined_features if we continue)
            iteration_record = RefinementIteration(
                iteration=iteration,
                suggestions=suggestion_response.suggestions,
                has_suggestions=suggestion_response.has_suggestions,
                reasoning=suggestion_response.reasoning,
                refined_caption=current_caption,
                changes_made=[]
            )
            
            # Check if we should stop
            if not suggestion_response.has_suggestions:
                if self.verbose:
                    print(f"\n✓ Converged after {iteration} iteration(s)")
                converged = True
                self.iterations.append(iteration_record)
                break
            
            # Step 2: Apply suggestions
            if self.verbose:
                print("  Editor: Applying suggestions...")
            
            refined_response = self.editor.refine(
                current_caption, 
                suggestion_response,
                transformed_content, 
                iteration
            )
            
            current_caption = refined_response["refined_caption"]
            
            # Update iteration record
            iteration_record.refined_caption = current_caption
            iteration_record.changes_made = refined_response["changes_made"]
            self.iterations.append(iteration_record)
            
            if self.verbose:
                print(f"  Editor: ✓ Applied {len(refined_response['changes_made'])} changes")
        
        # If we hit max iterations without converging
        if not converged and self.verbose:
            print(f"\n⚠ Reached maximum iterations ({self.max_iterations}) without convergence")
        
        # Prepare final result
        result = {
            "final_caption": current_caption,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "suggestions": it.suggestions,
                    "has_suggestions": it.has_suggestions,
                    "reasoning": it.reasoning,
                    "refined_caption": it.refined_caption,
                    "changes_made": it.changes_made
                }
                for it in self.iterations
            ],
            "converged": converged,
            "total_iterations": len(self.iterations),
            "convergence_iteration": len(self.iterations) if converged else None
        }
        
        return result
    
    def get_summary(self) -> str:
        """Get a summary of the refinement process"""
        if not self.iterations:
            return "No iterations performed yet"
        
        lines = [
            f"\n{'='*80}",
            "REFINEMENT SUMMARY",
            f"{'='*80}",
            f"Total iterations: {len(self.iterations)}",
            ""
        ]
        
        for it in self.iterations:
            lines.append(f"Iteration {it.iteration}:")
            lines.append(f"  - Suggestions: {len(it.suggestions)}")
            lines.append(f"  - Has suggestions: {it.has_suggestions}")
            lines.append(f"  - Reasoning: {it.reasoning}")
            lines.append(f"  - Changes made: {len(it.changes_made)}")
            
            if not it.has_suggestions:
                lines.append(f"  - ✓ CONVERGED")
        
        lines.append(f"{'='*80}")
        
        return '\n'.join(lines)


# Factory function for easy integration
def create_iterative_refinement_system(client, model: str, 
                                      max_iterations: int = 5,
                                      verbose: bool = True):
    """
    Create a complete iterative refinement system
    
    Args:
        client: Azure OpenAI client
        model: Model name
        max_iterations: Maximum refinement iterations
        verbose: Print progress
        
    Returns:
        IterativeRefinementController instance
    """
    suggester = SuggesterAgent(client, model, "Suggester")
    editor = EditorAgent(client, model, "Editor")
    
    return IterativeRefinementController(
        suggester, 
        editor, 
        max_iterations=max_iterations,
        verbose=verbose
    )


# Example usage
if __name__ == "__main__":
    from openai import AzureOpenAI
    import os
    
    # Setup
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Create system
    refinement_system = create_iterative_refinement_system(
        client, 
        "gpt-4o-mini",
        max_iterations=5,
        verbose=True
    )
    
    # Mock seed features
    seed_features = [
        {
            "focus_area": "objects",
            "features": "There are several vehicles including cars and trucks. Some pedestrians are visible."
        },
        {
            "focus_area": "spatial_relations",
            "features": "Objects are distributed around the ego vehicle. Some are close, others are far."
        },
        {
            "focus_area": "dynamics",
            "features": "Most vehicles appear to be moving. Pedestrians are walking."
        }
    ]
    
    # Run refinement
    result = refinement_system.refine(seed_features)
    
    # Print results
    print(refinement_system.get_summary())
    
    print(f"\n{'='*80}")
    print("FINAL REFINED FEATURES")
    print(f"{'='*80}")
    print(json.dumps(result['final_features'], indent=2))
    
    print(f"\n{'='*80}")
    print(f"Process {'CONVERGED' if result['converged'] else 'REACHED MAX ITERATIONS'}")
    if result['converged']:
        print(f"Converged at iteration: {result['convergence_iteration']}")
    print(f"{'='*80}")