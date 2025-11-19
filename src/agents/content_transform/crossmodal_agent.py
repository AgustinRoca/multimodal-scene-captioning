from typing import Any, Dict, List
from agents import BaseAgent
import json

class CrossModalAgent(BaseAgent):
    """Facilitates information sharing between agents"""
    
    def facilitate_exchange(self, agent_outputs: List[Dict]) -> Dict[str, Any]:
        """
        Enable agents to share and refine information
        
        Args:
            agent_outputs: List of outputs from all content transformation agents
        """
        system_prompt = """You are a coordination expert who facilitates information exchange.
Review outputs from multiple perception agents and:
- Identify complementary information
- Resolve contradictions
- Highlight important cross-modal insights
- Suggest areas needing attention
- Create summary of multi-modal understanding

Be concise and focused on actionable insights."""

        agent_summaries = {out["agent"]: out.get("observations", "") 
                          for out in agent_outputs}
        
        user_prompt = f"""Review and synthesize these agent observations:

{json.dumps(agent_summaries, indent=2)}

Provide a coordinated summary highlighting key insights and any discrepancies."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_llm(messages, temperature=0.4)
        
        return {
            "agent": self.agent_name,
            "modality": "cross_modal",
            "observations": response
        }