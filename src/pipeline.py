from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from openai import AzureOpenAI

from agents import *

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for Azure OpenAI models"""
    api_key: str
    endpoint: str
    api_version: str = "2025-01-01-preview"
    small_model: str = "gpt-4o-mini"  # For testing
    large_model: str = "gpt-4o"  # For production
    vision_model: str = "gpt-4o-mini"  # For vision tasks

@dataclass
class ModalityConfig:
    """Configuration for modality dropout"""
    use_cameras: bool = True
    use_lidar: bool = True
    use_annotations: bool = True
    camera_indices: Optional[List[int]] = None  # Which cameras to use (0-5)

# ============================================================================
# Main Pipeline
# ============================================================================

class SemanticCaptioningPipeline:
    """Complete semantic captioning pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        
        # Initialize all agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents in the pipeline"""
        # Layer 1: Content Transformation
        self.camera_agent = CameraAgent(
            self.client, self.config.vision_model, "CameraAgent"
        )
        self.lidar_agent = LiDARAgent(
            self.client, self.config.small_model, "LiDARAgent"
        )
        self.scene_graph_agent = SceneGraphAgent(
            self.client, self.config.small_model, "SceneGraphAgent"
        )
        self.cross_modal_agent = CrossModalAgent(
            self.client, self.config.small_model, "CrossModalAgent"
        )
        
        # Layer 2: Seed Features Generation
        self.seed_agent = SeedFeatureAgent(self.client, self.config.small_model)
        
        # Layer 3: Refinement
        self.suggester = SuggesterAgent(
            self.client, self.config.small_model, "SuggesterAgent"
        )
        self.editor = EditorAgent(
            self.client, self.config.small_model, "EditorAgent"
        )

        self.refinement_system = IterativeRefinementController(
            self.suggester, self.editor, max_iterations=3, verbose=False
        )
        
        # Layer 4: Caption Generation
        self.caption_generator = CaptionGenerator(
            self.client, self.config.small_model, "CaptionGenerator"
        )
    
    def process_scene(self, 
                     images: List[np.ndarray],
                     camera_names: List[str],
                     point_cloud: Optional[np.ndarray] = None,
                     annotations: Optional[List[Dict]] = None,
                     modality_config: ModalityConfig = None) -> Dict[str, Any]:
        """
        Process a complete scene through the pipeline
        
        Args:
            images: List of camera images
            camera_names: Names of camera views
            point_cloud: LiDAR point cloud data
            annotations: Object annotations
            modality_config: Configuration for modality dropout

        Returns:
            Complete pipeline output with structured caption
        """
        if modality_config is None:
            modality_config = ModalityConfig()
        
        results = {"pipeline_stages": {}}
        
        # Layer 1: Content Transformation with modality dropout
        print("Layer 1: Content Transformation...")
        layer1_outputs = []
        
        if modality_config.use_cameras and images:
            # Apply camera dropout if specified
            if modality_config.camera_indices:
                images = [images[i] for i in modality_config.camera_indices]
                camera_names = [camera_names[i] for i in modality_config.camera_indices]
            
            camera_output = self.camera_agent.process(images, camera_names)
            layer1_outputs.append(camera_output)
            print(f"  ✓ CameraAgent processed {len(images)} cameras")
        
        if modality_config.use_lidar and point_cloud is not None:
            lidar_output = self.lidar_agent.process(point_cloud)
            layer1_outputs.append(lidar_output)
            print(f"  ✓ LiDARAgent processed {len(point_cloud)} points")
        
        if modality_config.use_annotations and annotations:
            scene_graph_output = self.scene_graph_agent.process(annotations)
            layer1_outputs.append(scene_graph_output)
            print(f"  ✓ SceneGraphAgent processed {len(annotations)} objects")
        
        # Cross-modal coordination
        if len(layer1_outputs) > 1:
            cross_modal_output = self.cross_modal_agent.facilitate_exchange(layer1_outputs)
            layer1_outputs.append(cross_modal_output)
            print("  ✓ CrossModalAgent coordinated information")
        
        results["pipeline_stages"]["layer1_content_transformation"] = layer1_outputs
        
        # Layer 2: Seed Features Generation
        print("\nLayer 2: Seed Features Generation...")
        transformed_content = {
            "observations": [out.get("observations", "") for out in layer1_outputs]
        }
        
        seed_caption = self.seed_agent.generate_comprehensive_caption(transformed_content)
        
        results["pipeline_stages"]["layer2_seed_caption"] = seed_caption
        
        # Layer 3: Iterative Features Refinement
        print("\nLayer 3: Iterative Features Refinement...")
        
        refinement_result = self.refinement_system.refine(seed_caption['final_caption'], transformed_content)
        
        # Print summary
        convergence_status = "converged" if refinement_result['converged'] else "completed"
        print(f"  ✓ Refinement {convergence_status} after {refinement_result['total_iterations']} iteration(s)")
        
        # Store complete refinement history
        results["pipeline_stages"]["layer3_refinement"] = {
            "iterations": refinement_result['iterations'],
            "final_caption": refinement_result['final_caption'],
            "converged": refinement_result['converged'],
            "total_iterations": refinement_result['total_iterations'],
            "convergence_iteration": refinement_result.get('convergence_iteration')
        }
        
        # Prepare refined features for caption generation
        refined_caption = refinement_result['final_caption']
        
        # Layer 4: Caption Generation
        print("\nLayer 4: Caption Generation...")
        structured_caption = self.caption_generator.generate_structured_caption(
            refined_caption
        )
        print("  ✓ CaptionGenerator created structured caption")
        
        results["pipeline_stages"]["layer4_caption"] = structured_caption
        results["structured_caption"] = structured_caption["structured_caption"]
        
        # Add refinement metadata to results
        results["refinement_metadata"] = {
            "converged": refinement_result['converged'],
            "iterations": refinement_result['total_iterations']
        }
        
        return results
    
    def answer_mqa(self, question: str, scene_results: Dict) -> str:
        """Answer MQA question about a processed scene"""        
        structured_caption = scene_results["structured_caption"]
        
        return self.caption_generator.answer_mqa_question(
            question, structured_caption
        )
