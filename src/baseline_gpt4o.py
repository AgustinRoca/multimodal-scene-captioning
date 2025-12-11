"""
Raw GPT-4o Baseline for nuScenes-MQA
Two-step process for fair comparison with agentic pipeline:
1. Generate scene caption from cameras, LiDAR, and annotations
2. Answer questions using only the caption (no direct access to sensor data)
"""

import os
import sys
import base64
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from PIL import Image
from openai import AzureOpenAI

from nuscenes_loader import NuScenesLoader
from evaluation_framework import ComprehensiveMQAEvaluator


class RawGPT4oBaseline:
    """Baseline that uses raw GPT-4o without agentic flow"""
    
    def __init__(self, api_key: str, endpoint: str, model: str = "gpt-4o"):
        """
        Initialize baseline with OpenAI credentials
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint
            model: Model deployment name (default: gpt-4o)
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2025-01-01-preview",
            azure_endpoint=endpoint
        )
        self.model = model
        
    def encode_image(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def generate_scene_caption(self,
                              images: List[np.ndarray],
                              camera_names: List[str],
                              point_cloud: np.ndarray,
                              annotations: List[Dict]) -> str:
        """
        Generate a comprehensive scene caption from sensor data (Step 1)
        
        Args:
            images: List of camera images
            camera_names: Names of camera views
            point_cloud: LiDAR point cloud
            annotations: Object annotations
            
        Returns:
            Scene caption describing what's visible
        """
        # Prepare context from available data
        context_parts = []
        
        # Add point cloud statistics
        if point_cloud is not None and len(point_cloud) > 0:
            context_parts.append(self._describe_point_cloud(point_cloud))
        
        # Add annotations
        if annotations:
            context_parts.append(self._describe_annotations(annotations))
        
        # Add camera information
        if camera_names:
            context_parts.append(f"Available camera views: {', '.join(camera_names)}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt for caption generation
        system_prompt = """You are an expert at analyzing autonomous driving scenes from the nuScenes dataset.
You will be provided with:
- Multiple camera images from different viewpoints around a vehicle
- LiDAR point cloud statistics
- Object detection annotations

Your task is to generate a comprehensive, detailed caption describing the scene. Include:
1. What objects are visible and where they are located (by camera direction)
2. Counts of each type of object in different camera views
3. The overall environment and road conditions
4. Spatial relationships between objects

Be specific about:
- Object categories: adult pedestrian, child pedestrian, car, truck, bus, trailer, bicycle, motorcycle, barrier, traffic cone, construction vehicle
- Camera directions: front, front left, front right, back, back left, back right
- Object counts per camera view

This caption will be used to answer questions about the scene, so be thorough and precise."""
        
        user_prompt = f"""Scene Data:
{context}

Please generate a comprehensive caption describing this autonomous driving scene."""
        
        # Prepare messages with images
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add images as vision content
        user_content = []
        
        for img, cam_name in zip(images, camera_names):
            base64_image = self.encode_image(img)
            user_content.append({
                "type": "text",
                "text": f"Camera: {cam_name}"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"  # Use low detail to reduce tokens
                }
            })
        
        # Add text prompt
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Call GPT-4o
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=1000  # More tokens for detailed caption
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Error: Failed to generate scene caption"
    
    def answer_question_from_caption(self, question: str, scene_caption: str) -> str:
        """
        Answer MQA question based on scene caption only (Step 2)
        
        Args:
            question: MQA question to answer
            scene_caption: Previously generated scene caption
            
        Returns:
            Predicted answer in MQA format
        """
        system_prompt = """You are an expert at answering questions about autonomous driving scenes.

You will be given a scene caption and a question. Answer the question based ONLY on the information in the caption.

CRITICAL FORMATTING RULES:
1. For counting questions, use: <target><cnt>NUMBER</cnt> <obj>OBJECT_NAME</obj></target>
2. For binary questions, use: <ans>yes</ans> or <ans>no</ans>
3. For camera direction questions, use: <cam>DIRECTION</cam>
4. For distance questions, use: <dst>DISTANCE</dst>
5. For location questions, use: <loc>X, Y</loc>

IMPORTANT: 
- Object names should be singular (e.g., "car" not "cars")
- Use exact object categories from nuScenes: adult pedestrian, child pedestrian, car, truck, bus, trailer, bicycle, motorcycle, barrier, traffic cone, construction vehicle
- For camera directions use: front, front left, front right, back, back left, back right
- Only use information explicitly stated in the caption
- Be precise and concise in your answers

Examples:
Q: "How many cars are in the front camera?"
A: "<target><cnt>3</cnt> <obj>car</obj></target>"

Q: "Are there any pedestrians visible?"
A: "<ans>yes</ans>"

Q: "What objects are in the front left camera?"
A: "<target><cnt>2</cnt> <obj>car</obj></target>, <target><cnt>1</cnt> <obj>truck</obj></target>"
"""
        
        user_prompt = f"""Scene Caption:
{scene_caption}

Question: {question}

Please answer the question using the proper XML format based on the scene caption."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call GPT-4o
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"Error calling GPT-4o: {e}")
            return "<error>Failed to generate answer</error>"
    
    def _describe_point_cloud(self, point_cloud: np.ndarray) -> str:
        """Generate text description of point cloud"""
        num_points = len(point_cloud)
        
        if num_points == 0:
            return "LiDAR: No points detected"
        
        # Calculate basic statistics
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        
        description = f"""LiDAR Point Cloud Statistics:
- Total points: {num_points:,}
- X range: [{x.min():.1f}, {x.max():.1f}] m
- Y range: [{y.min():.1f}, {y.max():.1f}] m
- Z range: [{z.min():.1f}, {z.max():.1f}] m
- Average distance from ego: {np.sqrt(x**2 + y**2).mean():.1f} m"""
        
        return description
    
    def _describe_annotations(self, annotations: List[Dict]) -> str:
        """Generate text description of annotations"""
        if not annotations:
            return "Annotations: No objects detected"
        
        # Count objects by category
        from collections import Counter
        categories = Counter(ann['category_name'] for ann in annotations)
        
        # Group by general location (front, back, left, right)
        front_objects = []
        back_objects = []
        left_objects = []
        right_objects = []
        
        for ann in annotations:
            x, y = ann['translation'][0], ann['translation'][1]
            category = ann['category_name']
            
            # Classify by location (ego vehicle coordinate system)
            if x > 0:  # Front
                front_objects.append(category)
            else:  # Back
                back_objects.append(category)
                
            if y > 0:  # Left
                left_objects.append(category)
            else:  # Right
                right_objects.append(category)
        
        description = f"""Object Annotations:
- Total objects: {len(annotations)}
- Categories: {dict(categories)}
- Front region: {len(front_objects)} objects
- Back region: {len(back_objects)} objects
- Left region: {len(left_objects)} objects
- Right region: {len(right_objects)} objects"""
        
        return description


def run_baseline_evaluation(
    api_key: str,
    endpoint: str,
    nuscenes_dataroot: str,
    mqa_csv_path: str,
    nuscenes_version: str = "v1.0-mini",
    test_mode: bool = True,
    num_test_scenes: int = 5
):
    """
    Run complete baseline evaluation
    
    Args:
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint
        nuscenes_dataroot: Path to nuScenes dataset
        mqa_csv_path: Path to MQA CSV file
        nuscenes_version: nuScenes version to use
        test_mode: If True, only evaluate a subset
        num_test_scenes: Number of unique scenes (samples) for test mode
    """
    print("="*80)
    print("RAW GPT-4o BASELINE EVALUATION")
    print("="*80)
    
    # Initialize components
    print("\nInitializing...")
    baseline = RawGPT4oBaseline(api_key, endpoint)
    loader = NuScenesLoader(nuscenes_dataroot, nuscenes_version)
    evaluator = ComprehensiveMQAEvaluator(mqa_csv_path)
    
    # Get available samples
    scenes = loader.get_scene_list()
    available_sample_tokens = set()
    for scene in scenes:
        samples = loader.load_scene_samples(scene['token'])
        available_sample_tokens.update(s['sample_token'] for s in samples)
    
    print(f"Loaded {len(available_sample_tokens)} available samples")
    
    # Filter questions
    questions_df = evaluator.mqa_data[
        evaluator.mqa_data['sample_token'].isin(available_sample_tokens)
    ].copy()
    
    print(f"Filtered to {len(questions_df)} questions with available samples")
    
    if len(questions_df) == 0:
        print("ERROR: No matching samples found!")
        return None
    
    # Apply test mode filtering by unique scenes (sample_tokens)
    if test_mode:
        unique_samples = questions_df['sample_token'].unique()[:num_test_scenes]
        questions_df = questions_df[questions_df['sample_token'].isin(unique_samples)]
        print(f"\nTEST MODE: Evaluating {len(unique_samples)} scenes with {len(questions_df)} questions")
    else:
        unique_samples = questions_df['sample_token'].unique()
        print(f"\nFULL EVALUATION: Evaluating {len(unique_samples)} scenes with {len(questions_df)} questions")
    
    # Results storage
    all_results = []
    
    # Group questions by sample_token
    sample_groups = questions_df.groupby('sample_token')
    total_samples = len(sample_groups)
    
    for sample_idx, (sample_token, sample_questions) in enumerate(sample_groups):
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{total_samples}: {sample_token}")
        print(f"Questions for this sample: {len(sample_questions)}")
        print(f"{'='*80}")
        
        try:
            # Load sample once
            sample = loader.load_sample(sample_token)
            
            # STEP 1: Generate scene caption from all sensor data
            print("\n  Step 1: Generating scene caption...")
            scene_caption = baseline.generate_scene_caption(
                images=sample['images'],
                camera_names=sample['camera_names'],
                point_cloud=sample['point_cloud'],
                annotations=sample['annotations']
            )
            print(f"  Caption: {scene_caption[:150]}...")
            
            # STEP 2: Answer all questions for this sample using only the caption
            for q_idx, (_, question_row) in enumerate(sample_questions.iterrows()):
                question = question_row['question']
                print(f"\n  Question {q_idx + 1}/{len(sample_questions)}: {question[:60]}...")
                
                try:
                    # Get answer from caption only (fair comparison with agentic pipeline)
                    predicted_answer = baseline.answer_question_from_caption(
                        question=question,
                        scene_caption=scene_caption
                    )
                    
                    print(f"  Predicted: {predicted_answer[:100]}...")
                    
                    # Evaluate
                    try:
                        pred_parsed = evaluator.parse_tags_from_answer(predicted_answer)
                        gt_parsed = evaluator.parse_ground_truth_answer(question_row['answer'])
                        metrics = evaluator.compare_answers(pred_parsed['objects'], gt_parsed['objects'])
                    except Exception as eval_error:
                        print(f"  ERROR evaluating answer: {eval_error}")
                        metrics = {
                            'exact_match': 0.0,
                            'count_match': 0.0,
                            'object_match': 0.0,
                            'partial_credit': 0.0
                        }
                    
                    # Build result row
                    result_row = {
                        'sample_token': sample_token,
                        'question': question,
                        'ground_truth_answer': question_row['answer'],
                        'question_type': question_row['question_type'],
                        'scene_caption': scene_caption,
                        'predicted_answer': predicted_answer,
                        'exact_match': metrics['exact_match'],
                        'count_match': metrics['count_match'],
                        'object_match': metrics['object_match'],
                        'partial_credit': metrics['partial_credit']
                    }
                    
                    all_results.append(result_row)
                    
                    print(f"  Exact Match: {metrics['exact_match']}, "
                          f"Count: {metrics['count_match']:.2f}, "
                          f"Object: {metrics['object_match']:.2f}")
                    
                except Exception as e:
                    print(f"  ERROR answering question: {e}")
                    continue
                    
        except Exception as e:
            print(f"ERROR loading sample: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"baseline_gpt4o_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BASELINE GPT-4o EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"  Total Questions: {len(results_df)}")
    print(f"  Exact Match Accuracy: {results_df['exact_match'].mean():.2%}")
    print(f"  Count Accuracy: {results_df['count_match'].mean():.2%}")
    print(f"  Object Accuracy: {results_df['object_match'].mean():.2%}")
    
    print(f"\nPer Question Type:")
    for qtype, group in results_df.groupby('question_type'):
        print(f"  {qtype}:")
        print(f"    Count: {len(group)}")
        print(f"    Exact Match: {group['exact_match'].mean():.2%}")
        print(f"    Count Match: {group['count_match'].mean():.2%}")
        print(f"    Object Match: {group['object_match'].mean():.2%}")
    
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Configuration
    API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    NUSCENES_DATAROOT = "data/nuscenes/v1.0-mini"
    MQA_CSV_PATH = "data/nuscenes-mqa/df_train_mqa.csv"
    NUSCENES_VERSION = "v1.0-mini"
    
    # Run evaluation
    # Set test_mode=False for full evaluation across all scenes
    results = run_baseline_evaluation(
        api_key=API_KEY,
        endpoint=ENDPOINT,
        nuscenes_dataroot=NUSCENES_DATAROOT,
        mqa_csv_path=MQA_CSV_PATH,
        nuscenes_version=NUSCENES_VERSION,
        test_mode=True,
        num_test_scenes=10  # Number of unique scenes to evaluate
    )
