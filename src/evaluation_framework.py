"""
Complete nuScenes-MQA Evaluation System
Evaluates all modality combinations and generates detailed CSV results
"""

import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MQAMetrics:
    """Container for MQA evaluation metrics"""
    overall_accuracy: float
    per_question_type: Dict[str, float]
    count_accuracy: float
    object_accuracy: float
    total_questions: int
    correct_answers: int
    per_tag_accuracy: Dict[str, float]


class ComprehensiveMQAEvaluator:
    """Complete MQA evaluator with tag parsing and detailed metrics"""
    
    def __init__(self, mqa_csv_path: str):
        """Initialize evaluator with MQA dataset"""
        self.mqa_data = pd.read_csv(mqa_csv_path)
        print(f"Loaded {len(self.mqa_data)} MQA questions")
        print(f"Question types: {self.mqa_data['question_type'].value_counts().to_dict()}")
        
        # Object category mappings
        self.category_mappings = {
            'adult pedestrian': ['pedestrian', 'adult', 'person', 'people', 'adult pedestrian'],
            'child pedestrian': ['child', 'children'],
            'car': ['car', 'vehicle', 'automobile'],
            'truck': ['truck', 'trucks'],
            'bus': ['bus', 'buses'],
            'trailer': ['trailer', 'trailers'],
            'bicycle': ['bicycle', 'bike', 'cyclist', 'bicycles'],
            'motorcycle': ['motorcycle', 'motorbike', 'motorcycles'],
            'barrier': ['barrier', 'barriers'],
            'traffic cone': ['traffic cone', 'cone', 'traffic cones', 'cones'],
            'construction vehicle': ['construction', 'construction vehicle']
        }
        
        # Camera direction mappings
        self.camera_directions = {
            'front': ['front', 'forward'],
            'front left': ['front left', 'frontleft', 'front_left'],
            'front right': ['front right', 'frontright', 'front_right'],
            'back': ['back', 'rear', 'behind'],
            'back left': ['back left', 'backleft', 'back_left', 'rear left'],
            'back right': ['back right', 'backright', 'back_right', 'rear right']
        }
    
    def parse_tags_from_question(self, question: str) -> Dict[str, List[str]]:
        """
        Parse all XML tags from question
        
        Tags in nuScenes-MQA:
        - <obj>: Object (single word)
        - <cam>: Camera view
        - <dst>: Distance
        - <loc>: Location coordinates (x, y)
        
        Returns dict with lists of found tags
        """
        tags = {
            'obj': [],
            'cam': [],
            'dst': [],
            'loc': []
        }
        
        # Parse <obj> tags
        obj_matches = re.findall(r'<obj>(.*?)</obj>', question, re.IGNORECASE)
        tags['obj'] = [obj.strip() for obj in obj_matches]
        
        # Parse <cam> tags
        cam_matches = re.findall(r'<cam>(.*?)</cam>', question, re.IGNORECASE)
        tags['cam'] = [cam.strip() for cam in cam_matches]
        
        # Parse <dst> tags
        dst_matches = re.findall(r'<dst>(.*?)</dst>', question, re.IGNORECASE)
        tags['dst'] = [dst.strip() for dst in dst_matches]
        
        # Parse <loc> tags
        loc_matches = re.findall(r'<loc>(.*?)</loc>', question, re.IGNORECASE)
        tags['loc'] = [loc.strip() for loc in loc_matches]
        
        return tags
    
    def parse_tags_from_answer(self, answer: str) -> Dict[str, Any]:
        """
        Parse all tags from answer
        
        Tags in nuScenes-MQA answers:
        - <target>: Encapsulates <cnt> and <obj>
        - <cnt>: Count (single word/number)
        - <obj>: Object (single word)
        - <ans>: Binary response (single word)
        - <cam>: Camera view
        - <dst>: Distance
        - <loc>: Location coordinates
        
        Returns dict with parsed information
        """
        parsed = {
            'objects': [],  # List of {count, object} from <target> blocks
            'binary_answer': None,  # From <ans> tag
            'camera': None,  # From <cam> tag
            'distance': None,  # From <dst> tag
            'location': None  # From <loc> tag
        }
        
        # Find all <target>...</target> blocks
        target_pattern = r'<target>(.*?)</target>'
        targets = re.findall(target_pattern, answer, re.DOTALL | re.IGNORECASE)
        
        for target in targets:
            obj_dict = {}
            
            # Extract count from <cnt> tag
            cnt_match = re.search(r'<cnt>(\d+)</cnt>', target, re.IGNORECASE)
            if cnt_match:
                obj_dict['count'] = int(cnt_match.group(1))
            
            # Extract object from <obj> tag
            obj_match = re.search(r'<obj>(.*?)</obj>', target, re.IGNORECASE)
            if obj_match:
                obj_dict['object'] = obj_match.group(1).strip()
            
            if obj_dict:
                parsed['objects'].append(obj_dict)
        
        # Parse <ans> tag (binary answer)
        ans_match = re.search(r'<ans>(.*?)</ans>', answer, re.IGNORECASE)
        if ans_match:
            parsed['binary_answer'] = ans_match.group(1).strip().lower()
        
        # Parse <cam> tag
        cam_match = re.search(r'<cam>(.*?)</cam>', answer, re.IGNORECASE)
        if cam_match:
            parsed['camera'] = cam_match.group(1).strip()
        
        # Parse <dst> tag
        dst_match = re.search(r'<dst>(.*?)</dst>', answer, re.IGNORECASE)
        if dst_match:
            parsed['distance'] = dst_match.group(1).strip()
        
        # Parse <loc> tag
        loc_match = re.search(r'<loc>(.*?)</loc>', answer, re.IGNORECASE)
        if loc_match:
            parsed['location'] = loc_match.group(1).strip()
        
        return parsed
    
    def parse_ground_truth_answer(self, answer: str) -> List[Dict[str, Any]]:
        """Parse ground truth (handles multiple variations separated by :)"""
        first_variation = answer.split(':')[0]
        return self.parse_tags_from_answer(first_variation)
    
    def normalize_object_name(self, obj_name: str) -> str:
        """Normalize object name for comparison"""
        # Ensure obj_name is a string
        if not isinstance(obj_name, str):
            obj_name = str(obj_name)
        
        obj_name = obj_name.lower().strip()
        obj_name = obj_name.replace('_', ' ')
        obj_name = obj_name.replace('-', ' ')
        
        # Map to standard category
        for standard_name, variants in self.category_mappings.items():
            for variant in variants:
                if variant in obj_name or obj_name in variant:
                    return standard_name
        
        return obj_name
    
    def compare_answers(self, pred_objs: List[Dict], gt_objs: List[Dict]) -> Dict[str, float]:
        """Compare predicted and ground truth parsed answers"""
        metrics = {
            'exact_match': 0.0,
            'count_match': 0.0,
            'object_match': 0.0,
            'partial_credit': 0.0
        }
        
        if not pred_objs and not gt_objs:
            metrics['exact_match'] = 1.0
            metrics['count_match'] = 1.0
            metrics['object_match'] = 1.0
            return metrics
        
        if not pred_objs or not gt_objs:
            return metrics
        
        # Normalize objects
        pred_normalized = {}
        for obj in pred_objs:
            obj_name = self.normalize_object_name(obj.get('object', ''))
            count = obj.get('count', 0)
            pred_normalized[obj_name] = count
        
        gt_normalized = {}
        for obj in gt_objs:
            obj_name = self.normalize_object_name(obj.get('object', ''))
            count = obj.get('count', 0)
            gt_normalized[obj_name] = count
        
        # Exact match
        if pred_normalized == gt_normalized:
            metrics['exact_match'] = 1.0
            metrics['count_match'] = 1.0
            metrics['object_match'] = 1.0
            metrics['partial_credit'] = 1.0
            return metrics
        
        # Partial matching
        pred_objects = set(pred_normalized.keys())
        gt_objects = set(gt_normalized.keys())
        
        if pred_objects == gt_objects:
            metrics['object_match'] = 1.0
            count_matches = sum(1 for obj in gt_objects 
                              if pred_normalized.get(obj) == gt_normalized.get(obj))
            metrics['count_match'] = count_matches / len(gt_objects)
        else:
            overlap = pred_objects & gt_objects
            if overlap:
                metrics['object_match'] = len(overlap) / len(gt_objects)
                count_matches = sum(1 for obj in overlap 
                                  if pred_normalized.get(obj) == gt_normalized.get(obj))
                metrics['count_match'] = count_matches / len(gt_objects) if overlap else 0.0
        
        metrics['partial_credit'] = (metrics['object_match'] + metrics['count_match']) / 2
        
        return metrics
    
    def compute_metrics(self, results_df: pd.DataFrame) -> MQAMetrics:
        """Compute comprehensive metrics from results dataframe"""
        
        all_metrics = []
        metrics_by_type = defaultdict(list)
        metrics_by_tag = defaultdict(list)
        
        for idx, row in results_df.iterrows():
            pred_answer = row['predicted_answer']
            gt_answer = row['ground_truth_answer']
            question_type = row['question_type']
            question = row['question']
            
            try:
                # Parse answers
                pred_parsed = self.parse_tags_from_answer(pred_answer)
                gt_parsed = self.parse_ground_truth_answer(gt_answer)
                
                # Compare
                metrics = self.compare_answers(pred_parsed['objects'], gt_parsed['objects'])
                metrics['question_type'] = question_type
                
                all_metrics.append(metrics)
                metrics_by_type[question_type].append(metrics)
                
                # Group by tags in question
                question_tags = self.parse_tags_from_question(question)
                
                # Track metrics by object tags
                for obj_tag in question_tags['obj']:
                    normalized_obj = self.normalize_object_name(obj_tag)
                    metrics_by_tag[f"obj:{normalized_obj}"].append(metrics)
                
                # Track metrics by camera tags
                for cam_tag in question_tags['cam']:
                    metrics_by_tag[f"cam:{cam_tag}"].append(metrics)
                
                # Track metrics by distance tags
                for dst_tag in question_tags['dst']:
                    metrics_by_tag[f"dst:{dst_tag}"].append(metrics)
                
                # Track metrics by location tags
                for loc_tag in question_tags['loc']:
                    metrics_by_tag[f"loc:{loc_tag}"].append(metrics)
                    
            except Exception as e:
                print(f"\n⚠️  Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate
        if not all_metrics:
            return MQAMetrics(
                overall_accuracy=0.0,
                per_question_type={},
                count_accuracy=0.0,
                object_accuracy=0.0,
                total_questions=0,
                correct_answers=0,
                per_tag_accuracy={}
            )
        
        exact_matches = sum(m['exact_match'] for m in all_metrics)
        total = len(all_metrics)
        
        overall_accuracy = exact_matches / total
        count_accuracy = np.mean([m['count_match'] for m in all_metrics])
        object_accuracy = np.mean([m['object_match'] for m in all_metrics])
        
        # Per question type
        per_question_type = {}
        for qtype, metrics_list in metrics_by_type.items():
            exact = sum(m['exact_match'] for m in metrics_list)
            per_question_type[qtype] = {
                'accuracy': exact / len(metrics_list),
                'count_accuracy': np.mean([m['count_match'] for m in metrics_list]),
                'object_accuracy': np.mean([m['object_match'] for m in metrics_list]),
                'num_questions': len(metrics_list)
            }
        
        # Per tag
        per_tag_accuracy = {}
        for tag, metrics_list in metrics_by_tag.items():
            exact = sum(m['exact_match'] for m in metrics_list)
            per_tag_accuracy[tag] = {
                'accuracy': exact / len(metrics_list),
                'count': len(metrics_list)
            }
        
        return MQAMetrics(
            overall_accuracy=overall_accuracy,
            per_question_type=per_question_type,
            count_accuracy=count_accuracy,
            object_accuracy=object_accuracy,
            total_questions=total,
            correct_answers=int(exact_matches),
            per_tag_accuracy=per_tag_accuracy
        )
    
    def print_results(self, metrics: MQAMetrics):
        """Print formatted evaluation results"""
        print("\n" + "="*80)
        print("nuScenes-MQA EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nOverall Performance:")
        print(f"  Total Questions: {metrics.total_questions}")
        print(f"  Correct Answers: {metrics.correct_answers}")
        print(f"  Overall Accuracy: {metrics.overall_accuracy:.2%}")
        print(f"  Count Accuracy: {metrics.count_accuracy:.2%}")
        print(f"  Object Accuracy: {metrics.object_accuracy:.2%}")
        
        if metrics.per_question_type:
            print(f"\nPer Question Type:")
            print(f"{'Question Type':<50} {'Accuracy':<12} {'Count':<8}")
            print("-"*70)
            for qtype, qmetrics in sorted(metrics.per_question_type.items()):
                print(f"{qtype:<50} {qmetrics['accuracy']:<12.2%} {qmetrics['num_questions']:<8}")
        
        if metrics.per_tag_accuracy:
            print(f"\nPer Tag Performance (Top 20):")
            print(f"{'Tag':<40} {'Accuracy':<12} {'Count':<8}")
            print("-"*60)
            sorted_tags = sorted(metrics.per_tag_accuracy.items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:20]
            for tag, tag_metrics in sorted_tags:
                print(f"{tag:<40} {tag_metrics['accuracy']:<12.2%} {tag_metrics['count']:<8}")
        
        print("="*80)


class ModalityConfigGenerator:
    """Generates all possible modality configurations"""
    
    @staticmethod
    def generate_all_configs():
        """
        Generate all combinations of camera/lidar/annotations
        """
        from pipeline import ModalityConfig
        
        configs = {}
        
        # # Individual cameras
        # for i in range(1):
        #     configs[f'cam_{i}_only'] = ModalityConfig(
        #         use_cameras=True,
        #         camera_indices=[i],
        #         use_lidar=False,
        #         use_annotations=False
        #     )
        
        # Camera groups
        configs['all_cams'] = ModalityConfig(
            use_cameras=True,
            camera_indices=None,
            use_lidar=False,
            use_annotations=False
        )
        
        # configs['front_cams'] = ModalityConfig(
        #     use_cameras=True,
        #     camera_indices=[0, 1, 2],  # front, front_left, front_right
        #     use_lidar=False,
        #     use_annotations=False
        # )
        
        # LiDAR only
        configs['lidar_only'] = ModalityConfig(
            use_cameras=False,
            use_lidar=True,
            use_annotations=False
        )
        
        # # Annotations only
        # configs['annotations_only'] = ModalityConfig(
        #     use_cameras=False,
        #     use_lidar=False,
        #     use_annotations=True
        # )
        
        # Combinations
        configs['cams_lidar'] = ModalityConfig(
            use_cameras=True,
            use_lidar=True,
            use_annotations=False
        )
        
        configs['cams_annotations'] = ModalityConfig(
            use_cameras=True,
            use_lidar=False,
            use_annotations=True
        )
        
        # configs['lidar_annotations'] = ModalityConfig(
        #     use_cameras=False,
        #     use_lidar=True,
        #     use_annotations=True
        # )
        
        # Full (all modalities)
        configs['full'] = ModalityConfig(
            use_cameras=True,
            use_lidar=True,
            use_annotations=True
        )
        
        return configs


class ComprehensiveMQARunner:
    """Runs complete MQA evaluation with all modality combinations"""
    
    def __init__(self, pipeline, loader, mqa_csv_path: str):
        self.pipeline = pipeline
        self.loader = loader
        self.evaluator = ComprehensiveMQAEvaluator(mqa_csv_path)
        self.config_generator = ModalityConfigGenerator()
    
    def run_complete_evaluation(self,
                               test_mode: bool = False,
                               num_test_questions: int = 3,
                               output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Run evaluation on all modality combinations
        
        Args:
            test_mode: If True, only evaluate first num_test_questions
            num_test_questions: Number of questions to test
            output_csv: Path to save results CSV
            
        Returns:
            DataFrame with all results
        """
        
        # Get available sample tokens from the loader
        available_scenes = self.loader.get_scene_list()
        available_sample_tokens = set()
        
        print(f"Loading available sample tokens from nuScenes {self.loader.version}...")
        for scene in available_scenes:
            # Get all samples from this scene
            try:
                samples = self.loader.load_scene_samples(scene['token'], max_samples=None)
                for sample in samples:
                    available_sample_tokens.add(sample['sample_token'])
            except Exception as e:
                print(f"Warning: Could not load samples from scene {scene['name']}: {e}")
                continue
        
        print(f"Found {len(available_sample_tokens)} available sample tokens in dataset")
        
        # Filter MQA questions to only include available samples
        questions_df = self.evaluator.mqa_data[
            self.evaluator.mqa_data['sample_token'].isin(available_sample_tokens)
        ].copy()
        
        print(f"Filtered to {len(questions_df)} questions with available samples")
        
        if len(questions_df) == 0:
            print("ERROR: No matching samples found between MQA dataset and nuScenes version!")
            print(f"MQA has {len(self.evaluator.mqa_data)} questions")
            print(f"nuScenes {self.loader.version} has {len(available_sample_tokens)} samples")
            return pd.DataFrame()
        
        # Apply test mode filtering
        if test_mode:
            questions_df = questions_df.head(num_test_questions)
            print(f"\nTEST MODE: Evaluating {num_test_questions} questions")
        else:
            print(f"\nFULL EVALUATION: Evaluating {len(questions_df)} questions")
        
        # Generate modality configurations
        modality_configs = self.config_generator.generate_all_configs()
        print(f"Testing {len(modality_configs)} modality configurations")
        
        # Results storage
        all_results = []
        
        # Group questions by sample_token to avoid reprocessing
        sample_groups = questions_df.groupby('sample_token')
        
        total_samples = len(sample_groups)
        
        for sample_idx, (sample_token, sample_questions) in enumerate(sample_groups):
            print(f"\n{'='*80}")
            print(f"Sample {sample_idx + 1}/{total_samples}: {sample_token}")
            print(f"Questions for this sample: {len(sample_questions)}")
            print(f"{'='*80}")
            
            try:
                # Load sample once
                sample = self.loader.load_sample(sample_token)
                
                # Process with each modality configuration
                for config_name, modality_config in modality_configs.items():
                    print(f"\n  Config: {config_name}")
                    
                    try:
                        # Process scene
                        scene_result = self.pipeline.process_scene(
                            images=sample['images'],
                            camera_names=sample['camera_names'],
                            point_cloud=sample['point_cloud'],
                            annotations=sample['annotations'],
                            modality_config=modality_config
                        )
                        
                        # Get final caption
                        final_caption = scene_result['structured_caption']["full_caption"]
                        
                        # Answer all questions for this sample/config
                        for _, question_row in sample_questions.iterrows():
                            question = question_row['question']
                            
                            try:
                                # Get answer
                                predicted_answer = self.pipeline.answer_mqa(question, scene_result)
                                
                                # Build result row
                                result_row = {
                                    'sample_token': sample_token,
                                    'question': question,
                                    'ground_truth_answer': question_row['answer'],
                                    'question_type': question_row['question_type'],
                                    'config_name': config_name,
                                    'used_cam_0': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        0 in modality_config.camera_indices
                                    ),
                                    'used_cam_1': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        1 in modality_config.camera_indices
                                    ),
                                    'used_cam_2': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        2 in modality_config.camera_indices
                                    ),
                                    'used_cam_3': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        3 in modality_config.camera_indices
                                    ),
                                    'used_cam_4': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        4 in modality_config.camera_indices
                                    ),
                                    'used_cam_5': modality_config.use_cameras and (
                                        modality_config.camera_indices is None or 
                                        5 in modality_config.camera_indices
                                    ),
                                    'used_lidar': modality_config.use_lidar,
                                    'used_annotations': modality_config.use_annotations,
                                    'predicted_answer': predicted_answer,
                                    'final_scene_caption': final_caption
                                }
                                
                                all_results.append(result_row)
                                
                            except Exception as e:
                                print(f"Error answering question: {str(e)[:100]}")
                        
                    except Exception as e:
                        print(f"Error processing config {config_name}: {str(e)[:100]}")
                
            except Exception as e:
                print(f"Error loading sample {sample_token}: {str(e)}")
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add index
        results_df.insert(0, 'index', range(len(results_df)))
        
        # Save to CSV if specified
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and print comprehensive results"""
        
        if len(results_df) == 0:
            print("\nNo results to analyze - DataFrame is empty!")
            return
        
        print(f"\n{'='*80}")
        print("ANALYSIS: Overall Performance")
        print(f"{'='*80}")
        
        # Debug: Print first few rows
        print(f"\nDataFrame info:")
        print(f"  Rows: {len(results_df)}")
        print(f"  Columns: {list(results_df.columns)}")
        if 'config_name' in results_df.columns:
            print(f"  Configs: {results_df['config_name'].unique().tolist()}")
        else:
            print("'config_name' column not found!")
        
        # Overall metrics
        try:
            overall_metrics = self.evaluator.compute_metrics(results_df)
            self.evaluator.print_results(overall_metrics)
        except Exception as e:
            print(f"\nError computing overall metrics: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Per configuration analysis
        if 'config_name' not in results_df.columns:
            print("\nCannot analyze per-configuration: 'config_name' column missing")
            return
        
        print(f"\n{'='*80}")
        print("ANALYSIS: Per Configuration Performance")
        print(f"{'='*80}")
        
        config_results = []
        for config_name in results_df['config_name'].unique():
            config_df = results_df[results_df['config_name'] == config_name]
            
            try:
                config_metrics = self.evaluator.compute_metrics(config_df)
                
                config_results.append({
                    'config': config_name,
                    'accuracy': config_metrics.overall_accuracy,
                    'count_acc': config_metrics.count_accuracy,
                    'object_acc': config_metrics.object_accuracy,
                    'questions': config_metrics.total_questions
                })
            except Exception as e:
                print(f"  ⚠️  Error computing metrics for {config_name}: {e}")
                continue
        
        if not config_results:
            print("\nNo configuration results to display")
            return
        
        # Sort by accuracy
        config_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Count Acc':<12} {'Object Acc':<12} {'Questions':<10}")
        print("-"*70)
        for result in config_results:
            print(f"{result['config']:<25} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['count_acc']:<12.2%} "
                  f"{result['object_acc']:<12.2%} "
                  f"{result['questions']:<10}")


def main():
    """Main execution function"""
    from nuscenes_loader import create_loader
    from pipeline import SemanticCaptioningPipeline, ModelConfig
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # ========== CONFIGURATION ==========
    TEST_MODE = True  # Set to False for full evaluation
    NUM_TEST_QUESTIONS = 1  # Only used if TEST_MODE = True
    OUTPUT_DIR = "evaluation_results"
    # ===================================
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup pipeline
    config = ModelConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    
    pipeline = SemanticCaptioningPipeline(config)
    loader = create_loader(
        dataroot=os.getenv("NUSCENES_DATAROOT"),
        version=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
        use_mock=False
    )
    
    # MQA dataset path
    mqa_csv_path = "data/nuscenes-mqa/df_train_mqa.csv"
    
    # Initialize runner
    runner = ComprehensiveMQARunner(pipeline, loader, mqa_csv_path)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "test" if TEST_MODE else "full"
    output_csv = os.path.join(OUTPUT_DIR, f"mqa_results_{mode}_{timestamp}.csv")
    
    # Run evaluation
    print("\n" + "="*80)
    print("nuScenes-MQA COMPREHENSIVE EVALUATION")
    print("="*80)
    
    start_time = datetime.now()
    results_df = runner.run_complete_evaluation(
        test_mode=TEST_MODE,
        num_test_questions=NUM_TEST_QUESTIONS,
        output_csv=output_csv
    )
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEvaluation Duration: {duration}")
    
    # Analyze results
    runner.analyze_results(results_df)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_csv}")
    print(f"Total rows in CSV: {len(results_df)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()