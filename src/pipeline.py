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
    api_version: str = "2024-02-15-preview"
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
        self.lidar_agent = create_lidar_agent(
            self.client, self.config, "hybrid"
        )
        self.scene_graph_agent = SceneGraphAgent(
            self.client, self.config.small_model, "SceneGraphAgent"
        )
        self.bev_fusion_agent = BEVFusionAgent(
            self.client, self.config.small_model, "BEVFusionAgent"
        )
        self.cross_modal_agent = CrossModalAgent(
            self.client, self.config.small_model, "CrossModalAgent"
        )
        
        # Layer 2: Seed Features Generation
        self.seed_agents = [
            SeedFeatureAgent(self.client, self.config.small_model, 
                           f"SeedAgent_{area}", area)
            for area in ["objects", "scene_structure", "spatial_relations", 
                        "dynamics", "safety"]
        ]
        
        # Layer 3: Refinement
        self.suggester = SuggesterAgent(
            self.client, self.config.small_model, "SuggesterAgent"
        )
        self.editor = EditorAgent(
            self.client, self.config.small_model, "EditorAgent"
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
        
        # BEV Fusion (if we have camera and lidar)
        if modality_config.use_cameras and modality_config.use_lidar:
            cam_out = next((o for o in layer1_outputs if o["modality"] == "camera"), None)
            lid_out = next((o for o in layer1_outputs if o["modality"] == "lidar"), None)
            sg_out = next((o for o in layer1_outputs if o["modality"] == "scene_graph"), None)
            
            if cam_out and lid_out:
                bev_output = self.bev_fusion_agent.process(cam_out, lid_out, sg_out)
                layer1_outputs.append(bev_output)
                print("  ✓ BEVFusionAgent fused modalities")
        
        results["pipeline_stages"]["layer1_content_transformation"] = layer1_outputs
        
        # Layer 2: Seed Features Generation
        print("\nLayer 2: Seed Features Generation...")
        seed_features = []
        transformed_content = {
            "observations": [out.get("observations", "") for out in layer1_outputs]
        }
        
        for agent in self.seed_agents:
            seed_output = agent.generate(transformed_content)
            seed_features.append(seed_output)
            print(f"  ✓ {agent.agent_name} generated {agent.focus_area} features")
        
        results["pipeline_stages"]["layer2_seed_features"] = seed_features
        
        # Layer 3: Features Refinement
        print("\nLayer 3: Features Refinement...")
        suggestions = self.suggester.suggest(seed_features)
        print("  ✓ SuggesterAgent provided suggestions")
        
        refined_features = self.editor.refine(seed_features, suggestions)
        print("  ✓ EditorAgent refined features")
        
        results["pipeline_stages"]["layer3_refinement"] = {
            "suggestions": suggestions,
            "refined_features": refined_features
        }
        
        # Layer 4: Caption Generation
        print("\nLayer 4: Caption Generation...")
        structured_caption = self.caption_generator.generate_structured_caption(
            refined_features
        )
        print("  ✓ CaptionGenerator created structured caption")
        
        results["pipeline_stages"]["layer4_caption"] = structured_caption
        results["final_caption"] = structured_caption["structured_caption"]
        
        return results
    
    def answer_mqa(self, question: str, scene_results: Dict) -> str:
        """Answer MQA question about a processed scene"""
        refined_features = scene_results["pipeline_stages"]["layer3_refinement"]["refined_features"]
        structured_caption = scene_results["final_caption"]
        
        return self.caption_generator.answer_mqa_question(
            question, structured_caption, refined_features
        )
