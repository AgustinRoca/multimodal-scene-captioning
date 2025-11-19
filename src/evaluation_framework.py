import re
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any, Optional


@dataclass
class MQAMetrics:
    """Container for MQA evaluation metrics"""
    overall_accuracy: float
    per_question_type: Dict[str, float]
    count_accuracy: float
    object_accuracy: float
    direction_accuracy: float
    total_questions: int
    correct_answers: int


class NuScenesMQAEvaluator:
    """Evaluator for nuScenes-MQA benchmark"""
    
    def __init__(self, mqa_csv_path: Optional[str] = None):
        """
        Initialize MQA evaluator
        
        Args:
            mqa_csv_path: Path to df_train_mqa.csv or df_val_mqa.csv
        """
        self.mqa_data = None
        if mqa_csv_path:
            self.load_mqa_dataset(mqa_csv_path)
        
        # Object category mappings (nuScenes to common names)
        self.category_mappings = {
            'adult pedestrian': ['pedestrian', 'adult', 'person', 'people'],
            'car': ['car', 'vehicle', 'automobile'],
            'truck': ['truck'],
            'bus': ['bus'],
            'trailer': ['trailer'],
            'bicycle': ['bicycle', 'bike', 'cyclist'],
            'motorcycle': ['motorcycle', 'motorbike'],
            'barrier': ['barrier'],
            'traffic cone': ['traffic cone', 'cone'],
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
    
    def load_mqa_dataset(self, csv_path: str):
        """Load MQA dataset from CSV"""
        self.mqa_data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.mqa_data)} MQA questions from {csv_path}")
        print(f"Question types: {self.mqa_data['question_type'].value_counts().to_dict()}")
    
    def parse_mqa_answer(self, answer: str) -> List[Dict[str, Any]]:
        """
        Parse MQA format answer to extract all objects mentioned
        
        Example: "There are <target><cnt>2</cnt> <obj>cars</obj></target> and <target><cnt>1</cnt> <obj>pedestrian</obj></target>"
        Returns: [{'count': 2, 'object': 'cars'}, {'count': 1, 'object': 'pedestrian'}]
        """
        objects = []
        
        # Find all <target>...</target> blocks
        target_pattern = r'<target>(.*?)</target>'
        targets = re.findall(target_pattern, answer, re.DOTALL)
        
        for target in targets:
            obj_dict = {}
            
            # Extract count
            cnt_match = re.search(r'<cnt>(\d+)</cnt>', target)
            if cnt_match:
                obj_dict['count'] = int(cnt_match.group(1))
            
            # Extract object
            obj_match = re.search(r'<obj>(.*?)</obj>', target)
            if obj_match:
                obj_dict['object'] = obj_match.group(1).strip()
            
            if obj_dict:
                objects.append(obj_dict)
        
        # Also extract camera direction if present
        cam_match = re.search(r'<cam>(.*?)</cam>', answer)
        if cam_match:
            camera = cam_match.group(1).strip()
            for obj_dict in objects:
                obj_dict['camera'] = camera
        
        return objects
    
    def parse_ground_truth_answer(self, answer: str) -> List[Dict[str, Any]]:
        """
        Parse ground truth answer (may contain multiple variations separated by :)
        Takes the first variation for parsing
        """
        # Split by : to get first variation
        first_variation = answer.split(':')[0]
        return self.parse_mqa_answer(first_variation)
    
    def normalize_object_name(self, obj_name: str) -> str:
        """Normalize object name for comparison"""
        obj_name = obj_name.lower().strip()
        obj_name = obj_name.replace('_', ' ')
        obj_name = obj_name.replace('-', ' ')
        
        # Map to standard category
        for standard_name, variants in self.category_mappings.items():
            for variant in variants:
                if variant in obj_name or obj_name in variant:
                    return standard_name
        
        return obj_name
    
    def normalize_camera_direction(self, direction: str) -> str:
        """Normalize camera direction for comparison"""
        direction = direction.lower().strip()
        direction = direction.replace('_', ' ')
        
        for standard_dir, variants in self.camera_directions.items():
            if any(variant in direction or direction in variant for variant in variants):
                return standard_dir
        
        return direction
    
    def compare_objects(self, pred_objs: List[Dict], gt_objs: List[Dict]) -> Dict[str, float]:
        """
        Compare predicted objects with ground truth
        
        Returns metrics for this single answer
        """
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
        
        # Check exact match
        if pred_normalized == gt_normalized:
            metrics['exact_match'] = 1.0
            metrics['count_match'] = 1.0
            metrics['object_match'] = 1.0
            metrics['partial_credit'] = 1.0
            return metrics
        
        # Check object categories match (regardless of count)
        pred_objects = set(pred_normalized.keys())
        gt_objects = set(gt_normalized.keys())
        
        if pred_objects == gt_objects:
            metrics['object_match'] = 1.0
            
            # Check if counts match for matching objects
            count_matches = sum(1 for obj in gt_objects 
                              if pred_normalized.get(obj) == gt_normalized.get(obj))
            metrics['count_match'] = count_matches / len(gt_objects)
        else:
            # Partial credit for overlapping objects
            overlap = pred_objects & gt_objects
            if overlap:
                metrics['object_match'] = len(overlap) / len(gt_objects)
                
                # Check counts for overlapping objects
                count_matches = sum(1 for obj in overlap 
                                  if pred_normalized.get(obj) == gt_normalized.get(obj))
                metrics['count_match'] = count_matches / len(gt_objects) if overlap else 0.0
        
        # Partial credit: average of object and count match
        metrics['partial_credit'] = (metrics['object_match'] + metrics['count_match']) / 2
        
        return metrics
    
    def evaluate_single_answer(self, 
                              predicted_answer: str,
                              ground_truth_answer: str,
                              question_type: str) -> Dict[str, float]:
        """
        Evaluate a single MQA answer
        
        Args:
            predicted_answer: Model's answer
            ground_truth_answer: Ground truth answer from dataset
            question_type: Type of question
            
        Returns:
            Metrics for this answer
        """
        pred_objs = self.parse_mqa_answer(predicted_answer)
        gt_objs = self.parse_ground_truth_answer(ground_truth_answer)
        
        metrics = self.compare_objects(pred_objs, gt_objs)
        metrics['question_type'] = question_type
        
        return metrics
    
    def evaluate_batch(self, 
                      predictions: List[Dict[str, Any]],
                      use_loaded_dataset: bool = True) -> MQAMetrics:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of dicts with 'sample_token', 'question', 'predicted_answer'
            use_loaded_dataset: If True, match against loaded MQA dataset
            
        Returns:
            Aggregated MQA metrics
        """
        all_metrics = []
        metrics_by_type = defaultdict(list)
        
        for pred in predictions:
            sample_token = pred['sample_token']
            question = pred.get('question', '')
            predicted_answer = pred['predicted_answer']
            
            # Find ground truth
            if use_loaded_dataset and self.mqa_data is not None:
                # Match by sample_token and question
                matches = self.mqa_data[
                    (self.mqa_data['sample_token'] == sample_token) &
                    (self.mqa_data['question'] == question)
                ]
                
                if len(matches) == 0:
                    print(f"Warning: No ground truth found for sample {sample_token}")
                    continue
                
                gt_row = matches.iloc[0]
                ground_truth_answer = gt_row['answer']
                question_type = gt_row['question_type']
            else:
                # Use provided ground truth
                ground_truth_answer = pred.get('ground_truth_answer', '')
                question_type = pred.get('question_type', 'unknown')
            
            # Evaluate
            metrics = self.evaluate_single_answer(
                predicted_answer,
                ground_truth_answer,
                question_type
            )
            
            all_metrics.append(metrics)
            metrics_by_type[question_type].append(metrics)
        
        # Aggregate results
        if not all_metrics:
            return MQAMetrics(
                overall_accuracy=0.0,
                per_question_type={},
                count_accuracy=0.0,
                object_accuracy=0.0,
                direction_accuracy=0.0,
                total_questions=0,
                correct_answers=0
            )
        
        # Calculate overall metrics
        exact_matches = sum(m['exact_match'] for m in all_metrics)
        total = len(all_metrics)
        
        overall_accuracy = exact_matches / total
        count_accuracy = np.mean([m['count_match'] for m in all_metrics])
        object_accuracy = np.mean([m['object_match'] for m in all_metrics])
        
        # Calculate per question type accuracy
        per_question_type = {}
        for qtype, metrics_list in metrics_by_type.items():
            exact = sum(m['exact_match'] for m in metrics_list)
            per_question_type[qtype] = {
                'accuracy': exact / len(metrics_list),
                'count_accuracy': np.mean([m['count_match'] for m in metrics_list]),
                'object_accuracy': np.mean([m['object_match'] for m in metrics_list]),
                'partial_credit': np.mean([m['partial_credit'] for m in metrics_list]),
                'num_questions': len(metrics_list)
            }
        
        return MQAMetrics(
            overall_accuracy=overall_accuracy,
            per_question_type=per_question_type,
            count_accuracy=count_accuracy,
            object_accuracy=object_accuracy,
            direction_accuracy=0.0,  # Can be computed if needed
            total_questions=total,
            correct_answers=int(exact_matches)
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
            print(f"{'Question Type':<40} {'Accuracy':<12} {'Count Acc':<12} {'Object Acc':<12} {'Count':<8}")
            print("-"*80)
            
            for qtype, qmetrics in sorted(metrics.per_question_type.items()):
                print(f"{qtype:<40} "
                      f"{qmetrics['accuracy']:<12.2%} "
                      f"{qmetrics['count_accuracy']:<12.2%} "
                      f"{qmetrics['object_accuracy']:<12.2%} "
                      f"{qmetrics['num_questions']:<8}")
        
        print("="*80)


class MQAExperimentRunner:
    """Runs MQA experiments with the captioning pipeline"""
    
    def __init__(self, pipeline, loader, mqa_csv_path: str):
        """
        Initialize experiment runner
        
        Args:
            pipeline: SemanticCaptioningPipeline instance
            loader: NuScenesLoader instance
            mqa_csv_path: Path to MQA dataset CSV
        """
        self.pipeline = pipeline
        self.loader = loader
        self.evaluator = NuScenesMQAEvaluator(mqa_csv_path)
    
    def run_mqa_evaluation(self,
                          sample_tokens: Optional[List[str]] = None,
                          max_questions_per_sample: Optional[int] = None,
                          modality_config = None) -> Dict[str, Any]:
        """
        Run MQA evaluation on specified samples
        
        Args:
            sample_tokens: List of sample tokens to evaluate (None = all in MQA dataset)
            max_questions_per_sample: Limit questions per sample (None = all)
            modality_config: ModalityConfig for pipeline
            
        Returns:
            Results dictionary with predictions and metrics
        """
        # Get questions to evaluate
        if sample_tokens is None:
            # Use all unique samples from MQA dataset
            sample_tokens = self.evaluator.mqa_data['sample_token'].unique().tolist()
        
        print(f"Evaluating {len(sample_tokens)} samples...")
        
        all_predictions = []
        errors = []
        
        for i, sample_token in enumerate(sample_tokens):
            print(f"\nProcessing sample {i+1}/{len(sample_tokens)}: {sample_token}")
            
            try:
                # Load sample data
                sample = self.loader.load_sample(sample_token)
                
                # Process scene through pipeline
                scene_result = self.pipeline.process_scene(
                    images=sample['images'],
                    camera_names=sample['camera_names'],
                    point_cloud=sample['point_cloud'],
                    annotations=sample['annotations'],
                    modality_config=modality_config
                )
                
                # Get questions for this sample
                sample_questions = self.evaluator.mqa_data[
                    self.evaluator.mqa_data['sample_token'] == sample_token
                ]
                
                if max_questions_per_sample:
                    sample_questions = sample_questions.head(max_questions_per_sample)
                
                # Answer each question
                for _, row in sample_questions.iterrows():
                    question = row['question']
                    print(f"  Q: {question[:80]}...")
                    
                    # Get answer from pipeline
                    predicted_answer = self.pipeline.answer_mqa(question, scene_result)
                    print(f"  A: {predicted_answer[:80]}...")
                    
                    all_predictions.append({
                        'sample_token': sample_token,
                        'question': question,
                        'predicted_answer': predicted_answer,
                        'ground_truth_answer': row['answer'],
                        'question_type': row['question_type']
                    })
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                errors.append({
                    'sample_token': sample_token,
                    'error': str(e)
                })
        
        # Evaluate predictions
        print(f"\n\nEvaluating {len(all_predictions)} predictions...")
        metrics = self.evaluator.evaluate_batch(all_predictions, use_loaded_dataset=False)
        
        return {
            'predictions': all_predictions,
            'metrics': metrics,
            'errors': errors
        }
    
    def run_ablation_study(self,
                          sample_tokens: List[str],
                          modality_configs: Dict[str, Any],
                          max_questions_per_sample: Optional[int] = None) -> Dict[str, Any]:
        """
        Run ablation study with MQA evaluation
        
        Args:
            sample_tokens: List of sample tokens
            modality_configs: Dict of config_name -> ModalityConfig
            max_questions_per_sample: Limit questions per sample
            
        Returns:
            Results for each configuration
        """
        results = {}
        
        for config_name, modality_config in modality_configs.items():
            print(f"\n{'='*80}")
            print(f"Running configuration: {config_name}")
            print(f"{'='*80}")
            
            config_results = self.run_mqa_evaluation(
                sample_tokens=sample_tokens,
                max_questions_per_sample=max_questions_per_sample,
                modality_config=modality_config
            )
            
            results[config_name] = config_results
        
        return results
    
    def print_ablation_comparison(self, ablation_results: Dict[str, Any]):
        """Print comparison table of ablation study results"""
        print("\n" + "="*100)
        print("ABLATION STUDY - MQA ACCURACY COMPARISON")
        print("="*100)
        
        # Headers
        print(f"{'Configuration':<30} {'Overall Acc':<15} {'Count Acc':<15} "
              f"{'Object Acc':<15} {'Total Q':<10} {'Errors':<10}")
        print("-"*100)
        
        # Data rows
        for config_name, results in ablation_results.items():
            metrics = results['metrics']
            errors = len(results['errors'])
            
            print(f"{config_name:<30} "
                  f"{metrics.overall_accuracy:<15.2%} "
                  f"{metrics.count_accuracy:<15.2%} "
                  f"{metrics.object_accuracy:<15.2%} "
                  f"{metrics.total_questions:<10} "
                  f"{errors:<10}")
        
        print("="*100)
        
        # Detailed breakdown by question type
        print("\nDetailed Breakdown by Question Type:")
        for config_name, results in ablation_results.items():
            print(f"\n{config_name}:")
            self.evaluator.print_results(results['metrics'])


if __name__ == "__main__":
    from nuscenes_loader import create_loader
    from pipeline import SemanticCaptioningPipeline, ModelConfig, ModalityConfig
    import os
    from dotenv import load_dotenv
    
    load_dotenv()

    FULL_ABLATION_STUDY = True
    
    # Setup
    config = ModelConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    
    pipeline = SemanticCaptioningPipeline(config)
    loader = create_loader(
        dataroot=os.getenv("NUSCENES_DATAROOT"),
        version=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
        use_mock=False
    )
    
    # Path to MQA dataset
    mqa_csv_path = "data/nuscenes-mqa/df_train_mqa.csv"
    
    # Initialize runner
    runner = MQAExperimentRunner(pipeline, loader, mqa_csv_path)
    
    if not FULL_ABLATION_STUDY:
        print("Running MQA evaluation on first 3 samples...")
        sample_tokens = loader.get_scene_list()[0]['first_sample_token']
        results = runner.run_mqa_evaluation(
            sample_tokens=[sample_tokens],
            max_questions_per_sample=5  # Limit for testing
        )
        
        runner.evaluator.print_results(results['metrics'])
    
    else:
        print("\n\nRunning ablation study...")
        modality_configs = {
            'full': ModalityConfig(
                use_cameras=True,
                use_lidar=True,
                use_annotations=True
            ),
            'no_lidar': ModalityConfig(
                use_cameras=True,
                use_lidar=False,
                use_annotations=True
            ),
            'no_annotations': ModalityConfig(
                use_cameras=True,
                use_lidar=True,
                use_annotations=False
            ),
            'camera_only': ModalityConfig(
                use_cameras=True,
                use_lidar=False,
                use_annotations=False
            ),
        }
        
        # Get first 3 samples
        sample_tokens = [loader.get_scene_list()[i]['first_sample_token'] 
                        for i in range(min(3, len(loader.get_scene_list())))]
        
        ablation_results = runner.run_ablation_study(
            sample_tokens=sample_tokens,
            modality_configs=modality_configs,
            max_questions_per_sample=3  # Limit for testing
        )
        
        runner.print_ablation_comparison(ablation_results)
