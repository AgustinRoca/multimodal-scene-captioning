"""
Generate detailed logs of the complete pipeline transformation for documentation
Processes 3 sample scenes with full configuration and logs every agent output
"""

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from nuscenes_loader import NuScenesLoader
from pipeline import SemanticCaptioningPipeline, ModelConfig, ModalityConfig


def format_dict_pretty(d, indent=2):
    """Format dictionary for readable output"""
    return json.dumps(d, indent=indent, default=str)


def save_agent_output(file, agent_name, output_data, level=1):
    """Save agent output with formatting"""
    prefix = "  " * level
    file.write(f"\n{prefix}{'='*60}\n")
    file.write(f"{prefix}{agent_name}\n")
    file.write(f"{prefix}{'='*60}\n\n")
    
    if isinstance(output_data, dict):
        for key, value in output_data.items():
            file.write(f"{prefix}{key}:\n")
            if isinstance(value, (dict, list)):
                file.write(f"{prefix}  {format_dict_pretty(value, indent=2)}\n\n")
            else:
                file.write(f"{prefix}  {value}\n\n")
    else:
        file.write(f"{prefix}{output_data}\n\n")


def generate_detailed_logs(num_scenes=3):
    """
    Generate detailed logs for pipeline processing
    
    Args:
        num_scenes: Number of scenes to process
    """
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"pipeline_logs_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DETAILED PIPELINE LOGGING")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}")
    print(f"Processing {num_scenes} scenes with FULL configuration")
    print(f"{'='*80}\n")
    
    # Initialize pipeline with full configuration
    config = ModelConfig(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version="2025-01-01-preview",
        small_model="gpt-4o-mini",
        large_model="gpt-4o",
        vision_model="gpt-4o"
    )
    
    pipeline = SemanticCaptioningPipeline(config)
    
    # Load nuScenes data
    dataroot = "data/nuscenes/v1.0-mini"
    loader = NuScenesLoader(dataroot=dataroot, version='v1.0-mini')
    
    # Get first scene
    scenes = loader.get_scene_list()
    
    # Load samples from first scene
    
    samples = []
    for scene in scenes[:num_scenes]:
        print(f"Loading samples from scene: {scene['name']}")
        samples.extend(loader.load_scene_samples(scene['token'], max_samples=1))
    
    # Process each scene
    for i, sample_data in enumerate(samples):
        scene_num = i + 1
        sample_token = sample_data['sample_token']
        
        print(f"\n{'#'*80}")
        print(f"PROCESSING SCENE {scene_num}/{num_scenes}")
        print(f"Sample Token: {sample_token}")
        print(f"{'#'*80}\n")
        
        # Create log file for this scene
        log_file = output_dir / f"scene_{scene_num}_{sample_token[:8]}.log"
        
        # Create subdirectory for scene assets (images, annotations)
        scene_assets_dir = output_dir / f"scene_{scene_num}_{sample_token[:8]}_assets"
        scene_assets_dir.mkdir(exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"SCENE {scene_num} - COMPLETE PIPELINE TRANSFORMATION LOG\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Sample Token: {sample_token}\n")
            f.write(f"Timestamp: {sample_data['timestamp']}\n")
            f.write(f"Scene: {sample_data['scene_name']}\n")
            f.write(f"Description: {sample_data['scene_description']}\n")
            f.write(f"Location: {sample_data['metadata']['location']}\n\n")
            
            # Input data summary
            f.write(f"\n{'='*80}\n")
            f.write(f"INPUT DATA SUMMARY\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Camera Images: {len(sample_data['images'])}\n")
            for cam_name in sample_data['camera_names']:
                f.write(f"  - {cam_name}\n")
            f.write(f"\nLiDAR Points: {len(sample_data['point_cloud'])}\n")
            f.write(f"Annotations: {len(sample_data['annotations'])}\n")
            
            # List all annotated objects
            f.write(f"\nAnnotated Objects:\n")
            for ann in sample_data['annotations']:
                f.write(f"  - {ann['category_name']} at ({ann['translation'][0]:.1f}, {ann['translation'][1]:.1f}, {ann['translation'][2]:.1f})\n")
            
            # Full configuration (all modalities)
            modality_config = ModalityConfig(
                use_cameras=True,
                use_lidar=True,
                use_annotations=True,
                camera_indices=None  # Use all cameras
            )
            
            f.write(f"\n\n{'='*80}\n")
            f.write(f"PIPELINE CONFIGURATION\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Modality Configuration:\n")
            f.write(f"  - Cameras: {modality_config.use_cameras} (all {len(sample_data['images'])} cameras)\n")
            f.write(f"  - LiDAR: {modality_config.use_lidar}\n")
            f.write(f"  - Annotations: {modality_config.use_annotations}\n\n")
            
            # Save camera images
            print(f"  Saving camera images...")
            for img_array, cam_name in zip(sample_data['images'], sample_data['camera_names']):
                img = Image.fromarray(img_array)
                img_path = scene_assets_dir / f"{cam_name}.jpg"
                img.save(img_path)
            f.write(f"  Camera images saved to: {scene_assets_dir.name}/\n")
            
            # Save annotations to CSV
            print(f"  Saving annotations...")
            ann_data = []
            for ann in sample_data['annotations']:
                ann_data.append({
                    'category': ann['category_name'],
                    'instance_token': ann['instance_token'],
                    'x': ann['translation'][0],
                    'y': ann['translation'][1],
                    'z': ann['translation'][2],
                    'width': ann['size'][0],
                    'length': ann['size'][1],
                    'height': ann['size'][2],
                    'visibility': ann['visibility_token'],
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts']
                })
            df_annotations = pd.DataFrame(ann_data)
            csv_path = scene_assets_dir / "annotations.csv"
            df_annotations.to_csv(csv_path, index=False)
            f.write(f"  Annotations saved to: {scene_assets_dir.name}/annotations.csv\n\n")
            
            # Process the scene
            print(f"Processing scene {scene_num}...")
            results = pipeline.process_scene(
                images=sample_data['images'],
                camera_names=sample_data['camera_names'],
                point_cloud=sample_data['point_cloud'],
                annotations=sample_data['annotations'],
                modality_config=modality_config
            )
            
            # Log Layer 1: Content Transformation
            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER 1: CONTENT TRANSFORMATION\n")
            f.write(f"{'='*80}\n")
            
            layer1_outputs = results['pipeline_stages']['layer1_content_transformation']
            
            # Extract and save LiDAR BEV images if available
            print(f"  Extracting LiDAR visualizations...")
            for agent_output in layer1_outputs:
                agent_name = agent_output.get('agent', 'Unknown Agent')
                
                # Save LiDAR BEV images if this is the LiDAR agent
                if agent_name == 'LiDARAgent' and 'bev_metadata' in agent_output:
                    # The LiDAR agent should have stored BEV images internally
                    # We need to regenerate them for saving
                    # Find the lidar output to get BEV images
                    try:
                        # Re-process point cloud to get BEV visualizations
                        from agents.content_transform.lidar_agent import LiDARAgent
                        
                        # Create temporary agent just for BEV generation
                        class MockClient:
                            pass
                        temp_lidar_agent = LiDARAgent(MockClient(), "gpt-4o", "temp_lidar")
                        
                        # Process point cloud
                        processed_pc = temp_lidar_agent._preprocess_point_cloud(sample_data['point_cloud'])
                        ground_points, object_points = temp_lidar_agent._segment_ground(processed_pc)
                        bev_images = temp_lidar_agent._generate_multi_layer_bev(ground_points, object_points)
                        
                        # Save BEV visualizations
                        for layer_name, bev_img in bev_images.items():
                            img_path = scene_assets_dir / f"lidar_bev_{layer_name}.png"
                            cv2.imwrite(str(img_path), bev_img)
                        
                        f.write(f"  LiDAR BEV images saved to: {scene_assets_dir.name}/\n\n")
                        print(f"  Saved LiDAR BEV visualizations")
                    except Exception as e:
                        print(f"  Warning: Could not save LiDAR BEV images: {e}")
                        f.write(f"  LiDAR BEV images: Error saving - {e}\n\n")
                
                save_agent_output(f, agent_name, agent_output, level=1)
            
            # Log Layer 2: Seed Features Generation
            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER 2: SEED FEATURES GENERATION\n")
            f.write(f"{'='*80}\n\n")
            
            seed_caption = results['pipeline_stages']['layer2_seed_caption']
            save_agent_output(f, "SeedFeatureAgent", seed_caption, level=1)
            
            # Log Layer 3: Iterative Refinement
            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER 3: ITERATIVE FEATURES REFINEMENT\n")
            f.write(f"{'='*80}\n\n")
            
            refinement_data = results['pipeline_stages']['layer3_refinement']
            f.write(f"  Total Iterations: {refinement_data['total_iterations']}\n")
            f.write(f"  Converged: {refinement_data['converged']}\n")
            if refinement_data.get('convergence_iteration'):
                f.write(f"  Convergence at iteration: {refinement_data['convergence_iteration']}\n")
            f.write(f"\n")
            
            # Log each refinement iteration
            for iter_num, iteration in enumerate(refinement_data['iterations'], 1):
                f.write(f"\n  {'─'*60}\n")
                f.write(f"  ITERATION {iter_num}\n")
                f.write(f"  {'─'*60}\n\n")
                
                # Suggester output
                f.write(f"    SuggesterAgent Output:\n")
                f.write(f"      Suggestions: {iteration['suggestions']}\n\n")
                
                # Editor output
                f.write(f"    EditorAgent Output:\n")
                f.write(f"      Refined Caption:\n")
                f.write(f"      {iteration['refined_caption']}\n\n")
                
                # Convergence info
                if iteration.get('converged'):
                    f.write(f"      ✓ Convergence achieved at this iteration\n\n")
            
            f.write(f"\n  Final Refined Caption:\n")
            f.write(f"  {refinement_data['final_caption']}\n\n")
            
            # Log Layer 4: Caption Generation
            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER 4: CAPTION GENERATION\n")
            f.write(f"{'='*80}\n\n")
            
            caption_output = results['pipeline_stages']['layer4_caption']
            save_agent_output(f, "CaptionGenerator", caption_output, level=1)
            
            # Final structured caption
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL OUTPUT - STRUCTURED CAPTION\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{results['structured_caption']}\n\n")
            
            # Metadata
            f.write(f"\n{'='*80}\n")
            f.write(f"PIPELINE METADATA\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Refinement Converged: {results['refinement_metadata']['converged']}\n")
            f.write(f"Refinement Iterations: {results['refinement_metadata']['iterations']}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"END OF LOG\n")
            f.write(f"{'='*80}\n")
        
        print(f"  ✓ Scene {scene_num} logged to: {log_file}")
        print(f"  ✓ Scene {scene_num} assets saved to: {scene_assets_dir}")
    
    # Create summary file
    summary_file = output_dir / "SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"PIPELINE LOGGING SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Scenes Processed: {num_scenes}\n")
        f.write(f"Configuration: FULL (all cameras + LiDAR + annotations)\n\n")
        f.write(f"Log Files:\n")
        for i in range(1, num_scenes + 1):
            log_files = list(output_dir.glob(f"scene_{i}_*.log"))
            if log_files:
                f.write(f"  Scene {i}: {log_files[0].name}\n")
        f.write(f"\nAssets Directories:\n")
        for i in range(1, num_scenes + 1):
            asset_dirs = list(output_dir.glob(f"scene_{i}_*_assets"))
            if asset_dirs:
                f.write(f"  Scene {i}: {asset_dirs[0].name}/\n")
                f.write(f"    - 6 camera images (CAM_FRONT, CAM_BACK, etc.)\n")
                f.write(f"    - 3 LiDAR BEV images (semantic, height, density)\n")
                f.write(f"    - annotations.csv\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Pipeline Architecture:\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Layer 1: Content Transformation\n")
        f.write(f"  - CameraAgent: Processes all 6 camera images\n")
        f.write(f"  - LiDARAgent: Processes point cloud with BEV visualization\n")
        f.write(f"  - SceneGraphAgent: Processes object annotations\n")
        f.write(f"  - CrossModalAgent: Coordinates information across modalities\n\n")
        f.write(f"Layer 2: Seed Features Generation\n")
        f.write(f"  - SeedFeatureAgent: Generates initial comprehensive caption\n\n")
        f.write(f"Layer 3: Iterative Features Refinement\n")
        f.write(f"  - SuggesterAgent: Identifies improvement areas\n")
        f.write(f"  - EditorAgent: Refines caption based on suggestions\n")
        f.write(f"  - IterativeRefinementController: Manages refinement process\n\n")
        f.write(f"Layer 4: Caption Generation\n")
        f.write(f"  - CaptionGenerator: Creates final structured caption\n\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"LOGGING COMPLETE")
    print(f"{'='*80}\n")
    print(f"All logs saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"\nYou can now use these logs to create documentation/PDF with:")
    print(f"  - Complete agent outputs at each layer")
    print(f"  - Iterative refinement process")
    print(f"  - Final structured captions")
    print(f"  - Camera images for each scene")
    print(f"  - LiDAR BEV visualizations")
    print(f"  - Object annotations (CSV)")
    print(f"\n{'='*80}\n")


def main():
    """Main execution"""
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    num_scenes = 3
    if len(sys.argv) > 1:
        try:
            num_scenes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of scenes: {sys.argv[1]}")
            print("Usage: python generate_detailed_logs.py [num_scenes]")
            print("Example: python generate_detailed_logs.py 3")
            sys.exit(1)
    
    generate_detailed_logs(num_scenes)


if __name__ == "__main__":
    main()
