"""
Iterative Refinement System for Feature Enhancement
Implements suggest→edit loop until convergence or max iterations
"""

from typing import List, Dict, Any, Optional
import json
import re
from dataclasses import dataclass
from agents import BaseAgent, SuggesterAgent, EditorAgent


@dataclass
class RefinementIteration:
    """Records one iteration of refinement"""
    iteration: int
    suggestions: str
    refined_features: str
    has_suggestions: bool
    suggestion_count: int


class ConvergenceDetector:
    """Detects when refinement has converged (no more meaningful suggestions)"""
    
    def __init__(self):
        self.convergence_keywords = [
            "no further suggestions",
            "no additional improvements",
            "features are complete",
            "well-structured and comprehensive",
            "no significant issues",
            "ready for use",
            "all aspects are covered",
            "no major improvements needed"
        ]
        
        self.minimal_suggestion_patterns = [
            r"minor\s+(formatting|style|wording)",
            r"(optional|could consider|might)",
            r"no\s+(major|significant|critical)\s+issues"
        ]
    
    def has_converged(self, suggestions: str) -> bool:
        """
        Determine if suggestions indicate convergence
        
        Returns True if:
        - Suggestions explicitly state no improvements needed
        - Only minor/optional suggestions remain
        - Suggestions are very short (< 50 words)
        """
        suggestions_lower = suggestions.lower()
        
        # Check for explicit convergence keywords
        for keyword in self.convergence_keywords:
            if keyword in suggestions_lower:
                return True
        
        # Check for minimal suggestion patterns
        for pattern in self.minimal_suggestion_patterns:
            if re.search(pattern, suggestions_lower):
                # If multiple such patterns found, likely converged
                matches = sum(1 for p in self.minimal_suggestion_patterns 
                            if re.search(p, suggestions_lower))
                if matches >= 2:
                    return True
        
        # Check suggestion length (very short = likely converged)
        word_count = len(suggestions.split())
        if word_count < 50:
            return True
        
        return False
    
    def count_suggestions(self, suggestions: str) -> int:
        """
        Count the number of distinct suggestions
        
        Looks for:
        - Numbered lists (1., 2., 3.)
        - Bullet points (-, *, •)
        - Line breaks suggesting separate suggestions
        """
        # Count numbered items
        numbered = len(re.findall(r'\n\s*\d+\.', suggestions))
        
        # Count bullet points
        bullets = len(re.findall(r'\n\s*[-*•]', suggestions))
        
        # Use whichever is greater
        count = max(numbered, bullets)
        
        # If no clear markers, estimate by paragraphs
        if count == 0:
            paragraphs = [p.strip() for p in suggestions.split('\n\n') if p.strip()]
            count = len(paragraphs)
        
        return max(1, count)  # At least 1 if there's any text    

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
        self.convergence_detector = ConvergenceDetector()
        
        # Track refinement history
        self.iterations: List[RefinementIteration] = []
    
    def refine(self, seed_features: List[Dict]) -> Dict[str, Any]:
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
        
        # Convert seed features to dict format for processing
        current_features = {f["focus_area"]: f["features"] for f in seed_features}
        converged = False
        
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Step 1: Get suggestions
            if self.verbose:
                print("  Suggester: Analyzing features...")
            
            suggestions_result = self.suggester.suggest(current_features, iteration)
            suggestions = suggestions_result['suggestions']
            
            # Check for convergence
            has_converged = self.convergence_detector.has_converged(suggestions)
            suggestion_count = self.convergence_detector.count_suggestions(suggestions)
            
            if self.verbose:
                print(f"  Suggester: Found {suggestion_count} suggestions")
                if has_converged:
                    print("  Suggester: ✓ Convergence detected!")
            
            # Record iteration
            iteration_record = RefinementIteration(
                iteration=iteration,
                suggestions=suggestions,
                refined_features="",  # Will be filled if we continue
                has_suggestions=not has_converged,
                suggestion_count=suggestion_count
            )
            
            # Check if we should stop
            if has_converged:
                if self.verbose:
                    print(f"\n✓ Converged after {iteration} iteration(s)")
                converged = True
                iteration_record.refined_features = json.dumps(current_features, indent=2)
                self.iterations.append(iteration_record)
                break
            
            # Step 2: Apply suggestions
            if self.verbose:
                print("  Editor: Applying suggestions...")
            
            refined_result = self.editor.refine(
                current_features, 
                suggestions_result, 
                iteration
            )
            
            refined_features_text = refined_result['refined_features']
            
            # Try to parse refined features back to dict
            try:
                # Clean markdown code blocks if present
                cleaned = refined_features_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                
                current_features = json.loads(cleaned)
                
            except json.JSONDecodeError:
                # If parsing fails, keep as text and wrap in structure
                if self.verbose:
                    print("  Warning: Could not parse refined features as JSON, keeping as text")
                current_features = {"refined_text": refined_features_text}
            
            iteration_record.refined_features = refined_features_text
            self.iterations.append(iteration_record)
            
            if self.verbose:
                print(f"  Editor: ✓ Features refined")
        
        # If we hit max iterations without converging
        if not converged and self.verbose:
            print(f"\n⚠ Reached maximum iterations ({self.max_iterations}) without convergence")
        
        # Prepare final result
        result = {
            "final_features": current_features,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "suggestions": it.suggestions,
                    "refined_features": it.refined_features,
                    "has_suggestions": it.has_suggestions,
                    "suggestion_count": it.suggestion_count
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
            lines.append(f"  - Suggestions: {it.suggestion_count}")
            lines.append(f"  - Has meaningful suggestions: {it.has_suggestions}")
            
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
        api_version="2024-02-15-preview",
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