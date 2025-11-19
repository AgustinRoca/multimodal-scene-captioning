import re
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    exact_match: float
    additional_metrics: Dict[str, float]


class NuScenesMQAEvaluator:
    """Evaluator for nuScenes-MQA style questions"""
    
    def __init__(self):
        self.object_categories = [
            'car', 'truck', 'bus', 'pedestrian', 'bicycle', 
            'motorcycle', 'trailer', 'barrier', 'traffic_cone'
        ]
        
        # Visibility bins
        self.visibility_bins = {
            '0-40%': (0, 40),
            '40-60%': (40, 60),
            '60-80%': (60, 80),
            '80-100%': (80, 100)
        }
        
        # Direction bins (degrees from ego vehicle)
        self.direction_bins = {
            'front': (315, 45),
            'front_right': (45, 135),
            'back_right': (135, 225),
            'back': (225, 315),
            'front_left': (225, 315),
            'back_left': (225, 315)
        }
    
    def parse_mqa_answer(self, answer: str) -> Dict[str, Any]:
        """
        Parse MQA format answer
        
        Example: "There are <target><cnt>2</cnt> <obj>cars</obj></target>."
        Returns: {'count': 2, 'object': 'cars'}
        """
        parsed = {}
        
        # Extract count
        cnt_match = re.search(r'<cnt>(\d+)</cnt>', answer)
        if cnt_match:
            parsed['count'] = int(cnt_match.group(1))
        
        # Extract object
        obj_match = re.search(r'<obj>(\w+)</obj>', answer)
        if obj_match:
            parsed['object'] = obj_match.group(1)
        
        # Extract location/direction
        cam_match = re.search(r'<cam>(\w+)</cam>', answer)
        if cam_match:
            parsed['camera'] = cam_match.group(1)
        
        return parsed
    
    def evaluate_counting_question(self, 
                                   predicted_answer: str,
                                   ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate counting question
        
        Args:
            predicted_answer: Model's answer in MQA format
            ground_truth: Ground truth with 'count' and 'object' fields
        """
        parsed = self.parse_mqa_answer(predicted_answer)
        
        metrics = {
            'exact_match': 0.0,
            'count_accuracy': 0.0,
            'object_match': 0.0
        }
        
        if 'count' in parsed and 'count' in ground_truth:
            # Count accuracy (allow ±1 tolerance)
            count_diff = abs(parsed['count'] - ground_truth['count'])
            if count_diff == 0:
                metrics['count_accuracy'] = 1.0
                metrics['exact_match'] = 1.0
            elif count_diff == 1:
                metrics['count_accuracy'] = 0.5
        
        if 'object' in parsed and 'object' in ground_truth:
            # Normalize object names
            pred_obj = parsed['object'].lower().replace('_', '').replace('.', '')
            gt_obj = ground_truth['object'].lower().replace('_', '').replace('.', '')
            
            if pred_obj == gt_obj or pred_obj in gt_obj or gt_obj in pred_obj:
                metrics['object_match'] = 1.0
        
        return metrics
    
    def evaluate_batch(self, 
                      predictions: List[str],
                      ground_truths: List[Dict[str, Any]],
                      question_types: List[str]) -> EvaluationMetrics:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of model answers
            ground_truths: List of ground truth annotations
            question_types: List of question types ('counting', 'existence', etc.)
        """
        all_metrics = defaultdict(list)
        
        for pred, gt, qtype in zip(predictions, ground_truths, question_types):
            if qtype == 'counting':
                metrics = self.evaluate_counting_question(pred, gt)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
        
        # Aggregate metrics
        aggregated = {}
        for key, values in all_metrics.items():
            aggregated[key] = np.mean(values)
        
        return EvaluationMetrics(
            accuracy=aggregated.get('count_accuracy', 0.0),
            precision=aggregated.get('object_match', 0.0),
            recall=aggregated.get('object_match', 0.0),
            f1_score=aggregated.get('object_match', 0.0),
            exact_match=aggregated.get('exact_match', 0.0),
            additional_metrics=aggregated
        )


class StructuredCaptionEvaluator:
    """Evaluator for structured JSON captions"""
    
    def __init__(self):
        self.required_fields = [
            'scene_summary',
            'ego_vehicle',
            'objects',
            'road_structure',
            'environment'
        ]
    
    def evaluate_structure(self, caption: Dict) -> Dict[str, float]:
        """Evaluate if caption has correct structure"""
        metrics = {
            'has_all_required_fields': 1.0,
            'field_completeness': 0.0
        }
        
        # Check required fields
        for field in self.required_fields:
            if field not in caption:
                metrics['has_all_required_fields'] = 0.0
                break
        
        # Calculate field completeness
        present_fields = sum(1 for f in self.required_fields if f in caption)
        metrics['field_completeness'] = present_fields / len(self.required_fields)
        
        return metrics
    
    def evaluate_object_detection(self, 
                                  predicted_caption: Dict,
                                  ground_truth_annotations: List[Dict]) -> Dict[str, float]:
        """
        Evaluate object detection accuracy
        
        Args:
            predicted_caption: Generated caption with objects
            ground_truth_annotations: nuScenes annotations
        """
        metrics = {
            'object_count_error': 0.0,
            'category_precision': 0.0,
            'category_recall': 0.0,
            'category_f1': 0.0
        }
        
        if 'objects' not in predicted_caption:
            return metrics
        
        # Count objects by category in ground truth
        gt_categories = defaultdict(int)
        for ann in ground_truth_annotations:
            category = self._normalize_category(ann['category_name'])
            gt_categories[category] += 1
        
        # Count objects by category in prediction
        pred_categories = defaultdict(int)
        for obj in predicted_caption['objects']:
            category = self._normalize_category(obj.get('category', ''))
            pred_categories[category] += 1
        
        # Calculate count error
        total_error = sum(abs(pred_categories[cat] - gt_categories[cat]) 
                         for cat in set(list(pred_categories.keys()) + list(gt_categories.keys())))
        metrics['object_count_error'] = total_error
        
        # Calculate precision, recall, F1 for categories
        all_categories = set(list(pred_categories.keys()) + list(gt_categories.keys()))
        
        true_positives = sum(min(pred_categories[cat], gt_categories[cat]) 
                           for cat in all_categories)
        predicted_total = sum(pred_categories.values())
        actual_total = sum(gt_categories.values())
        
        metrics['category_precision'] = (true_positives / predicted_total 
                                        if predicted_total > 0 else 0.0)
        metrics['category_recall'] = (true_positives / actual_total 
                                     if actual_total > 0 else 0.0)
        
        if metrics['category_precision'] + metrics['category_recall'] > 0:
            metrics['category_f1'] = (2 * metrics['category_precision'] * metrics['category_recall'] / 
                                     (metrics['category_precision'] + metrics['category_recall']))
        
        return metrics
    
    @staticmethod
    def _normalize_category(category: str) -> str:
        """Normalize category name"""
        # Remove nuScenes prefixes
        category = category.lower()
        category = category.replace('vehicle.', '')
        category = category.replace('human.pedestrian.', 'pedestrian_')
        category = category.replace('movable_object.', '')
        category = category.replace('static_object.', '')
        return category
    
    def evaluate_visibility_binning(self,
                                   predicted_objects: List[Dict],
                                   ground_truth_annotations: List[Dict]) -> float:
        """Evaluate if objects are in correct visibility bins"""
        correct = 0
        total = 0
        
        # Create mapping of objects by rough position
        for pred_obj in predicted_objects:
            if 'visibility' not in pred_obj:
                continue
            
            pred_vis = pred_obj['visibility']
            # Find matching ground truth object (simplified matching by category)
            matching_gt = None
            for gt_ann in ground_truth_annotations:
                if self._categories_match(pred_obj.get('category', ''), 
                                         gt_ann['category_name']):
                    matching_gt = gt_ann
                    break
            
            if matching_gt:
                gt_vis = matching_gt['visibility_token']
                if self._visibility_bins_match(pred_vis, gt_vis):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def _categories_match(cat1: str, cat2: str) -> bool:
        """Check if two categories match (fuzzy)"""
        cat1 = cat1.lower().replace('_', '').replace('.', '')
        cat2 = cat2.lower().replace('_', '').replace('.', '')
        return cat1 in cat2 or cat2 in cat1
    
    @staticmethod
    def _visibility_bins_match(vis1: str, vis2: str) -> bool:
        """Check if visibility bins match"""
        # Extract percentages
        def extract_range(s):
            matches = re.findall(r'(\d+)', s)
            if len(matches) >= 2:
                return int(matches[0]), int(matches[1])
            return None
        
        range1 = extract_range(vis1)
        range2 = extract_range(vis2)
        
        if range1 and range2:
            # Check if ranges overlap
            return not (range1[1] < range2[0] or range2[1] < range1[0])
        
        return False


class ExperimentRunner:
    """Runs ablation experiments and compares results"""
    
    def __init__(self, pipeline, loader):
        self.pipeline = pipeline
        self.loader = loader
        self.mqa_evaluator = NuScenesMQAEvaluator()
        self.caption_evaluator = StructuredCaptionEvaluator()
    
    def run_ablation_study(self, 
                          sample_tokens: List[str],
                          modality_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ablation study with different modality configurations
        
        Args:
            sample_tokens: List of sample tokens to process
            modality_configs: Dict of config_name -> ModalityConfig
            
        Returns:
            Results for each configuration
        """
        results = {}
        
        for config_name, modality_config in modality_configs.items():
            print(f"\n{'='*80}")
            print(f"Running configuration: {config_name}")
            print(f"{'='*80}")
            
            config_results = {
                'captions': [],
                'processing_times': [],
                'errors': []
            }
            
            for i, sample_token in enumerate(sample_tokens):
                print(f"Processing sample {i+1}/{len(sample_tokens)}...")
                
                try:
                    # Load sample
                    sample = self.loader.load_sample(sample_token)
                    
                    # Process through pipeline
                    import time
                    start_time = time.time()
                    
                    result = self.pipeline.process_scene(
                        images=sample['images'],
                        camera_names=sample['camera_names'],
                        point_cloud=sample['point_cloud'],
                        annotations=sample['annotations'],
                        modality_config=modality_config
                    )
                    
                    processing_time = time.time() - start_time
                    
                    config_results['captions'].append({
                        'sample_token': sample_token,
                        'caption': result['final_caption'],
                        'ground_truth_annotations': sample['annotations']
                    })
                    config_results['processing_times'].append(processing_time)
                    
                except Exception as e:
                    print(f"Error processing sample {sample_token}: {e}")
                    config_results['errors'].append(str(e))
            
            results[config_name] = config_results
        
        return results
    
    def evaluate_results(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate results from ablation study"""
        evaluation_summary = {}
        
        for config_name, config_results in ablation_results.items():
            print(f"\nEvaluating configuration: {config_name}")
            
            # Evaluate structured captions
            structure_scores = []
            object_detection_scores = []
            
            for caption_data in config_results['captions']:
                caption = caption_data['caption']
                gt_annotations = caption_data['ground_truth_annotations']
                
                # Structure evaluation
                struct_metrics = self.caption_evaluator.evaluate_structure(caption)
                structure_scores.append(struct_metrics)
                
                # Object detection evaluation
                obj_metrics = self.caption_evaluator.evaluate_object_detection(
                    caption, gt_annotations
                )
                object_detection_scores.append(obj_metrics)
            
            # Aggregate scores
            evaluation_summary[config_name] = {
                'avg_processing_time': np.mean(config_results['processing_times']),
                'structure_completeness': np.mean([s['field_completeness'] 
                                                   for s in structure_scores]),
                'object_count_mae': np.mean([s['object_count_error'] 
                                            for s in object_detection_scores]),
                'object_category_f1': np.mean([s['category_f1'] 
                                              for s in object_detection_scores]),
                'num_errors': len(config_results['errors'])
            }
        
        return evaluation_summary
    
    def print_comparison_table(self, evaluation_summary: Dict[str, Any]):
        """Print comparison table of different configurations"""
        print("\n" + "="*100)
        print("ABLATION STUDY RESULTS")
        print("="*100)
        
        # Headers
        print(f"{'Configuration':<30} {'Proc. Time (s)':<15} {'Structure':<12} "
              f"{'Obj Count MAE':<15} {'Obj Cat F1':<12} {'Errors':<10}")
        print("-"*100)
        
        # Data rows
        for config_name, metrics in evaluation_summary.items():
            print(f"{config_name:<30} "
                  f"{metrics['avg_processing_time']:<15.2f} "
                  f"{metrics['structure_completeness']:<12.2%} "
                  f"{metrics['object_count_mae']:<15.2f} "
                  f"{metrics['object_category_f1']:<12.2%} "
                  f"{metrics['num_errors']:<10}")
        
        print("="*100)


if __name__ == "__main__":
    from nuscenes_loader import create_loader
    from pipeline import SemanticCaptioningPipeline, ModelConfig, ModalityConfig
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Setup
    config = ModelConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    
    pipeline = SemanticCaptioningPipeline(config)
    loader = create_loader(os.getenv("NUSCENES_DATAROOT"), os.getenv("NUSCENES_VERSION", "v1.0-mini"), use_mock=False)

    # Define ablation configurations
    modality_configs = {
        'full': ModalityConfig(
            use_cameras=True,
            use_lidar=True,
            use_annotations=True
        ),
        'camera_only': ModalityConfig(
            use_cameras=True,
            use_lidar=False,
            use_annotations=False
        ),
        'no_lidar': ModalityConfig(
            use_cameras=True,
            use_lidar=False,
            use_annotations=True
        ),
        'front_camera_only': ModalityConfig(
            use_cameras=True,
            use_lidar=False,
            use_annotations=False,
            camera_indices=[0]  # Only front camera
        )
    }
    
    # Run experiments
    runner = ExperimentRunner(pipeline, loader)
    
    # Get some sample tokens
    sample_tokens = [scene['first_sample_token'] for scene in loader.get_scene_list()[:3]]
    
    print("Running ablation study...")
    results = runner.run_ablation_study(sample_tokens, modality_configs)
    
    print("\nEvaluating results...")
    evaluation = runner.evaluate_results(results)
    
    runner.print_comparison_table(evaluation)