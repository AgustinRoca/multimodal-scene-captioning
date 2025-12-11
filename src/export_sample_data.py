"""
Export sample data for visualization and analysis
Exports camera images, LiDAR BEV visualizations, and annotations for a specific sample
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

from nuscenes_loader import NuScenesLoader
from agents.content_transform.lidar_agent import LiDARAgent


def export_sample_data(sample_token: str, output_dir: str):
    """
    Export all data for a specific sample
    
    Args:
        sample_token: nuScenes sample token
        output_dir: Directory to save exported data
    """
    # Initialize
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting data for sample: {sample_token}")
    print(f"Output directory: {output_path}")
    
    # Load nuScenes data
    dataroot = "data/nuscenes/v1.0-mini"
    loader = NuScenesLoader(dataroot=dataroot, version='v1.0-mini')
    
    # Get sample data
    print("\nLoading sample data...")
    sample_data = loader.load_sample(sample_token)
    
    # Export camera images
    print("\nExporting camera images...")
    for i, (img_array, cam_name) in enumerate(zip(sample_data['images'], sample_data['camera_names'])):
        img = Image.fromarray(img_array)
        img_path = output_path / f"{cam_name}.jpg"
        img.save(img_path)
        print(f"  Saved: {img_path}")
    
    # Export LiDAR visualizations
    print("\nProcessing LiDAR data...")
    point_cloud = sample_data['point_cloud']
    
    # Initialize LiDAR agent (no actual LLM client needed for visualization)
    class MockClient:
        pass
    
    lidar_agent = LiDARAgent(MockClient(), "gpt-4o", "lidar_agent")
    
    # Preprocess point cloud
    processed_pc = lidar_agent._preprocess_point_cloud(point_cloud)
    
    # Segment ground and objects
    ground_points, object_points = lidar_agent._segment_ground(processed_pc)
    
    # Generate BEV visualizations
    bev_images = lidar_agent._generate_multi_layer_bev(ground_points, object_points)
    
    print("\nExporting LiDAR BEV visualizations...")
    for layer_name, bev_img in bev_images.items():
        if layer_name == 'semantic':
            # RGB image
            img_path = output_path / f"lidar_bev_{layer_name}.png"
            cv2.imwrite(str(img_path), bev_img)
            print(f"  Saved: {img_path}")
        else:
            # Grayscale maps (height, density)
            img_path = output_path / f"lidar_bev_{layer_name}.png"
            cv2.imwrite(str(img_path), bev_img)
            print(f"  Saved: {img_path}")
    
    # Export annotations to CSV
    print("\nExporting annotations...")
    annotations = sample_data['annotations']
    
    # Convert annotations to DataFrame
    ann_data = []
    for ann in annotations:
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
    csv_path = output_path / "annotations.csv"
    df_annotations.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Total annotations: {len(df_annotations)}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    print(f"Sample Token: {sample_token}")
    print(f"\nCamera Images: {len(sample_data['images'])}")
    for cam_name in sample_data['camera_names']:
        print(f"  - {cam_name}")
    
    print(f"\nLiDAR Data:")
    print(f"  Total points: {len(point_cloud)}")
    print(f"  Ground points: {len(ground_points)}")
    print(f"  Object points: {len(object_points)}")
    
    print(f"\nAnnotations: {len(annotations)}")
    print("\nObject counts by category:")
    category_counts = df_annotations['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    print("\n" + "="*80)
    print(f"All data exported to: {output_path}")
    print("="*80 + "\n")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python export_sample_data.py <sample_token> [output_dir]")
        print("\nExample: python export_sample_data.py ca9a282c9e77460f8360f564131a8af5")
        print("         python export_sample_data.py ca9a282c9e77460f8360f564131a8af5 exported_data/sample1")
        sys.exit(1)
    
    sample_token = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"exported_data/{sample_token}"
    
    export_sample_data(sample_token, output_dir)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
