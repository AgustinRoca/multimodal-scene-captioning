"""
MQA Evaluation Results Visualization
Generates comprehensive graphs and tables from evaluation CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

# Set style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (16, 16)
plt.rcParams['font.size'] = 64
plt.rcParams['axes.linewidth'] = 0  # Remove plot border
plt.rcParams['axes.edgecolor'] = 'none'  # Remove plot border color


class MQAResultsVisualizer:
    """Visualize MQA evaluation results from CSV"""
    
    def __init__(self, csv_path: str, baseline_csv_path: str = None):
        """Load results from CSV"""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(csv_path).parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load baseline if provided
        self.baseline_df = None
        self.baseline_accuracy = None
        if baseline_csv_path:
            try:
                self.baseline_df = pd.read_csv(baseline_csv_path)
                # Calculate baseline accuracy
                if 'exact_match' in self.baseline_df.columns:
                    self.baseline_accuracy = self.baseline_df['exact_match'].mean()
                else:
                    pred_parsed = self.baseline_df['predicted_answer'].apply(self.parse_answer)
                    gt_parsed = self.baseline_df['ground_truth_answer'].apply(
                        lambda x: self.parse_answer(x.split(':')[0])
                    )
                    correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
                    self.baseline_accuracy = correct / len(self.baseline_df)
                print(f"Loaded baseline: {len(self.baseline_df)} results, accuracy: {self.baseline_accuracy:.2%}")
            except Exception as e:
                print(f"Warning: Could not load baseline from {baseline_csv_path}: {e}")
        
        print(f"Loaded {len(self.df)} evaluation results")
    
    def parse_tags_from_question(self, question: str) -> dict:
        """Parse XML tags from question"""
        tags = {
            'obj': [],
            'cam': [],
            'dst': [],
            'loc': []
        }
        
        tags['obj'] = re.findall(r'<obj>(.*?)</obj>', question, re.IGNORECASE)
        tags['cam'] = re.findall(r'<cam>(.*?)</cam>', question, re.IGNORECASE)
        tags['dst'] = re.findall(r'<dst>(.*?)</dst>', question, re.IGNORECASE)
        tags['loc'] = re.findall(r'<loc>(.*?)</loc>', question, re.IGNORECASE)
        
        return tags
    
    def parse_answer(self, answer: str) -> dict:
        """Simple answer parsing for comparison"""
        parsed = {
            'objects': [],
            'binary': None
        }
        
        # Parse <target> blocks
        targets = re.findall(r'<target>(.*?)</target>', answer, re.DOTALL | re.IGNORECASE)
        for target in targets:
            cnt_match = re.search(r'<cnt>(\d+)</cnt>', target, re.IGNORECASE)
            obj_match = re.search(r'<obj>(.*?)</obj>', target, re.IGNORECASE)
            
            if cnt_match and obj_match:
                parsed['objects'].append({
                    'count': int(cnt_match.group(1)),
                    'object': obj_match.group(1).strip().lower()
                })
        
        # Parse <ans> tag
        ans_match = re.search(r'<ans>(.*?)</ans>', answer, re.IGNORECASE)
        if ans_match:
            parsed['binary'] = ans_match.group(1).strip().lower()
        
        return parsed
    
    def print_overall_performance(self):
        """Print overall accuracy statistics"""
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        
        pred_parsed = self.df['predicted_answer'].apply(self.parse_answer)
        gt_parsed = self.df['ground_truth_answer'].apply(
            lambda x: self.parse_answer(x.split(':')[0])
        )
        correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
        total = len(self.df)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nAgentic Pipeline:")
        print(f"  Total Evaluations: {total}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {total - correct}")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        
        if self.baseline_accuracy is not None:
            print(f"\nBaseline GPT-4o:")
            print(f"  Total Evaluations: {len(self.baseline_df)}")
            print(f"  Overall Accuracy: {self.baseline_accuracy:.2%}")
            improvement = (accuracy - self.baseline_accuracy) / self.baseline_accuracy * 100 if self.baseline_accuracy > 0 else 0
            print(f"\nImprovement over baseline: {accuracy - self.baseline_accuracy:+.2%} ({improvement:+.1f}%)")
        
        print("="*80 + "\n")
    
    def create_merged_scene_table(self):
        """Create merged table with questions per scene and question type distribution"""
        print("="*80)
        print("QUESTIONS PER SCENE WITH QUESTION TYPE DISTRIBUTION")
        print("="*80 + "\n")
        
        # Get unique questions per scene
        scene_data = []
        for sample_token in self.df['sample_token'].unique():
            scene_df = self.df[self.df['sample_token'] == sample_token]
            
            # Get unique questions and question types
            num_questions = scene_df['question'].nunique()
            num_question_types = scene_df['question_type'].nunique()
            
            # Get question type counts for this scene
            qtype_counts = scene_df.groupby('question_type')['question'].nunique().to_dict()
            
            row = {
                'Sample Token': sample_token,
                'Num Questions': num_questions,
                'Num Question Types': num_question_types
            }
            
            # Add question type columns
            for qtype, count in qtype_counts.items():
                row[qtype] = count
            
            scene_data.append(row)
        
        # Create DataFrame
        scene_df = pd.DataFrame(scene_data)
        
        # Sort by number of questions
        scene_df = scene_df.sort_values('Num Questions', ascending=False)
        
        # Fill NaN with 0 for question types
        for col in scene_df.columns:
            if col not in ['Sample Token', 'Num Questions', 'Num Question Types']:
                scene_df[col] = scene_df[col].fillna(0).astype(int)
        
        # Print table
        print(scene_df.to_string(index=False))
        
        # Print totals
        print("\n" + "-"*80)
        print("TOTALS:")
        total_questions = self.df['question'].nunique()
        print(f"  Total Unique Questions: {total_questions}")
        
        # Question type distribution across all data
        qtype_distribution = self.df.groupby('question_type')['question'].nunique()
        for qtype, count in qtype_distribution.items():
            print(f"  {qtype}: {count}")
        
        print("="*80 + "\n")
    
    def create_question_type_table(self):
        """Create accuracy table by question type"""
        print("="*80)
        print("ACCURACY BY QUESTION TYPE")
        print("="*80 + "\n")
        
        qtype_data = []
        
        for qtype in self.df['question_type'].unique():
            qtype_df = self.df[self.df['question_type'] == qtype]
            
            pred_parsed = qtype_df['predicted_answer'].apply(self.parse_answer)
            gt_parsed = qtype_df['ground_truth_answer'].apply(
                lambda x: self.parse_answer(x.split(':')[0])
            )
            
            correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
            total = len(qtype_df)
            accuracy = correct / total if total > 0 else 0
            
            row = {
                'Question Type': qtype,
                'Accuracy': f"{accuracy:.2%}",
                'Correct': correct,
                'Total': total
            }
            
            # Add baseline comparison if available
            if self.baseline_df is not None:
                baseline_qtype_df = self.baseline_df[self.baseline_df['question_type'] == qtype]
                if len(baseline_qtype_df) > 0:
                    if 'exact_match' in baseline_qtype_df.columns:
                        baseline_acc = baseline_qtype_df['exact_match'].mean()
                    else:
                        b_pred = baseline_qtype_df['predicted_answer'].apply(self.parse_answer)
                        b_gt = baseline_qtype_df['ground_truth_answer'].apply(
                            lambda x: self.parse_answer(x.split(':')[0])
                        )
                        baseline_correct = sum(p == g for p, g in zip(b_pred, b_gt))
                        baseline_acc = baseline_correct / len(baseline_qtype_df)
                    row['Baseline'] = f"{baseline_acc:.2%}"
                    row['Δ vs Baseline'] = f"{accuracy - baseline_acc:+.2%}"
            
            qtype_data.append(row)
        
        # Create DataFrame and sort by total
        qtype_df = pd.DataFrame(qtype_data)
        qtype_df = qtype_df.sort_values('Total', ascending=False)
        
        print(qtype_df.to_string(index=False))
        print("="*80 + "\n")
    
    def plot_config_comparison(self):
        """Compare accuracy across different configurations"""
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Calculate accuracy per config
        config_accuracy = {}
        for config in self.df['config_name'].unique():
            config_df = self.df[self.df['config_name'] == config]
            
            pred_parsed = config_df['predicted_answer'].apply(self.parse_answer)
            gt_parsed = config_df['ground_truth_answer'].apply(
                lambda x: self.parse_answer(x.split(':')[0])
            )
            
            correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
            accuracy = correct / len(config_df) if len(config_df) > 0 else 0
            config_accuracy[config] = accuracy
        
        # Add baseline if available
        all_configs = list(config_accuracy.items())
        if self.baseline_accuracy is not None:
            all_configs.append(('Baseline GPT-4o', self.baseline_accuracy))
        
        # Sort by accuracy
        configs = sorted(all_configs, key=lambda x: x[1], reverse=True)
        config_names = [c[0] for c in configs]
        accuracies = [c[1] for c in configs]
        
        # Format config names: replace _ with space and capitalize (except for baseline)
        formatted_names = []
        colors = []
        for name, acc in configs:
            if name == 'Baseline GPT-4o':
                formatted_names.append('Baseline\nGPT-4o')
                colors.append('#e74c3c')  # Red for baseline
            else:
                formatted = name.replace('_', ' ').title()
                # Split long labels
                if formatted == 'Cams Annotations':
                    formatted = 'Cams\nAnnotations'
                formatted_names.append(formatted)
                colors.append(None)  # Will use viridis colormap
        
        # Create color array
        if self.baseline_accuracy is not None:
            # Use custom colors: baseline is red, others use viridis
            final_colors = []
            viridis_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(configs) - 1))
            viridis_idx = 0
            for color in colors:
                if color is None:
                    final_colors.append(viridis_colors[viridis_idx])
                    viridis_idx += 1
                else:
                    final_colors.append(color)
        else:
            final_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(configs)))
        
        # Create bar plot
        bars = ax.barh(formatted_names, accuracies, color=final_colors)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.2%}', va='center', fontweight='bold', fontsize=64)
        
        ax.set_xlabel('Accuracy', fontsize=72, fontweight='bold')
        ax.set_ylabel('Configuration', fontsize=72, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=64)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Remove x-axis ticks
        ax.set_xlim(0, max(accuracies) * 1.15 if accuracies else 1)
        
        plt.tight_layout()
        output_path = self.output_dir / "config_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_object_tags_top10(self):
        """Plot top 10 object tags by frequency"""
        fig, ax = plt.subplots(figsize=(18, 18))
        
        # Extract object tags and compute accuracy
        tag_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for idx, row in self.df.iterrows():
            tags = self.parse_tags_from_question(row['question'])
            
            pred = self.parse_answer(row['predicted_answer'])
            gt = self.parse_answer(row['ground_truth_answer'].split(':')[0])
            is_correct = (pred == gt)
            
            for tag in tags['obj']:
                tag_accuracy[tag]['total'] += 1
                if is_correct:
                    tag_accuracy[tag]['correct'] += 1
        
        if not tag_accuracy:
            print("No object tags found")
            return
        
        # Calculate accuracy percentages
        tag_results = {
            tag: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for tag, stats in tag_accuracy.items()
        }
        
        # Sort by frequency and get top 10
        sorted_tags = sorted(tag_accuracy.items(), 
                           key=lambda x: x[1]['total'], reverse=True)[:10]
        
        tags = [t[0] for t in sorted_tags]
        accuracies = [tag_results[t[0]] for t in sorted_tags]
        counts = [t[1]['total'] for t in sorted_tags]
        
        # Plot
        bars = ax.barh(tags, accuracies, color=plt.cm.coolwarm(np.array(accuracies)))
        
        # Add labels
        for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
            ax.text(acc + 0.01, i, f'{acc:.1%} (n={count})', 
                   va='center', fontsize=56, fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=72, fontweight='bold')
        ax.set_ylabel('Object Tag', fontsize=72, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=64)
        ax.set_xlim(0, max(accuracies) * 1.2 if accuracies else 1)
        
        plt.tight_layout()
        output_path = self.output_dir / "object_tags_top10.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_camera_tags_top10(self):
        """Plot top 10 camera tags by frequency"""
        fig, ax = plt.subplots(figsize=(18, 18))
        
        # Extract camera tags and compute accuracy
        tag_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for idx, row in self.df.iterrows():
            tags = self.parse_tags_from_question(row['question'])
            
            pred = self.parse_answer(row['predicted_answer'])
            gt = self.parse_answer(row['ground_truth_answer'].split(':')[0])
            is_correct = (pred == gt)
            
            for tag in tags['cam']:
                tag_accuracy[tag]['total'] += 1
                if is_correct:
                    tag_accuracy[tag]['correct'] += 1
        
        if not tag_accuracy:
            print("No camera tags found")
            return
        
        # Calculate accuracy percentages
        tag_results = {
            tag: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for tag, stats in tag_accuracy.items()
        }
        
        # Sort by frequency and get top 10
        sorted_tags = sorted(tag_accuracy.items(), 
                           key=lambda x: x[1]['total'], reverse=True)[:10]
        
        tags = [t[0] for t in sorted_tags]
        accuracies = [tag_results[t[0]] for t in sorted_tags]
        counts = [t[1]['total'] for t in sorted_tags]
        
        # Plot
        bars = ax.barh(tags, accuracies, color=plt.cm.coolwarm(np.array(accuracies)))
        
        # Add labels
        for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
            ax.text(acc + 0.01, i, f'{acc:.1%} (n={count})', 
                   va='center', fontsize=56, fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=72, fontweight='bold')
        ax.set_ylabel('Camera Tag', fontsize=72, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=64)
        ax.set_xlim(0, max(accuracies) * 1.2 if accuracies else 1)
        
        plt.tight_layout()
        output_path = self.output_dir / "camera_tags_top10.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_baseline_vs_best_config(self):
        """Plot direct comparison between baseline and best configuration"""
        if self.baseline_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Get best config accuracy
        config_accuracy = {}
        for config in self.df['config_name'].unique():
            config_df = self.df[self.df['config_name'] == config]
            pred_parsed = config_df['predicted_answer'].apply(self.parse_answer)
            gt_parsed = config_df['ground_truth_answer'].apply(
                lambda x: self.parse_answer(x.split(':')[0])
            )
            correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
            accuracy = correct / len(config_df) if len(config_df) > 0 else 0
            config_accuracy[config] = accuracy
        
        best_config = max(config_accuracy.items(), key=lambda x: x[1])
        best_config_name = best_config[0].replace('_', ' ').title()
        best_accuracy = best_config[1]
        
        # Create comparison bars
        systems = ['Baseline\nGPT-4o', f'Best Config\n({best_config_name})']
        accuracies = [self.baseline_accuracy, best_accuracy]
        colors = ['#e74c3c', '#27ae60']
        
        bars = ax.bar(systems, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.2%}',
                   ha='center', va='bottom', fontsize=56, fontweight='bold')
        
        # Add improvement annotation
        improvement = best_accuracy - self.baseline_accuracy
        improvement_pct = (improvement / self.baseline_accuracy * 100) if self.baseline_accuracy > 0 else 0
        
        # Draw arrow showing improvement
        arrow_y = min(accuracies) + (max(accuracies) - min(accuracies)) / 2
        ax.annotate('', xy=(1, best_accuracy), xytext=(0, self.baseline_accuracy),
                   arrowprops=dict(arrowstyle='->', lw=8, color='blue', alpha=0.5))
        ax.text(0.5, arrow_y, f'+{improvement:.2%}\n({improvement_pct:+.1f}%)',
               ha='center', va='center', fontsize=48, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_ylabel('Accuracy', fontsize=72, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=64)
        ax.set_ylim(0, max(accuracies) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "baseline_vs_best.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_question_type_accuracy(self):
        """Plot accuracy by question type"""
        fig, ax = plt.subplots(figsize=(18, 18))
        
        # Calculate accuracy per question type
        qtype_data = []
        for qtype in self.df['question_type'].unique():
            qtype_df = self.df[self.df['question_type'] == qtype]
            
            pred_parsed = qtype_df['predicted_answer'].apply(self.parse_answer)
            gt_parsed = qtype_df['ground_truth_answer'].apply(
                lambda x: self.parse_answer(x.split(':')[0])
            )
            
            correct = sum(p == g for p, g in zip(pred_parsed, gt_parsed))
            total = len(qtype_df)
            accuracy = correct / total if total > 0 else 0
            
            qtype_data.append({
                'type': qtype,
                'accuracy': accuracy,
                'total': total
            })
        
        # Sort by total count
        qtype_data = sorted(qtype_data, key=lambda x: x['total'], reverse=True)
        
        # Format question type names
        formatted_qtypes = []
        for d in qtype_data:
            qtype = d['type'].replace('_', ' ').capitalize()
            # Split long labels into two lines
            if qtype == 'Important object count and direction':
                qtype = 'Important object count\nand direction'
            elif qtype == 'Object presence confirmation':
                qtype = 'Object presence\nconfirmation'
            formatted_qtypes.append(qtype)
        
        qtypes = formatted_qtypes
        accuracies = [d['accuracy'] for d in qtype_data]
        totals = [d['total'] for d in qtype_data]
        
        # Plot
        bars = ax.barh(qtypes, accuracies, color=plt.cm.coolwarm(np.array(accuracies)))
        
        # Add labels
        for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
            ax.text(acc + 0.01, i, f'{acc:.1%} (n={total})', 
                   va='center', fontsize=56, fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=72, fontweight='bold')
        ax.set_ylabel('Question Type', fontsize=72, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=64)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Remove x-axis ticks
        ax.set_xlim(0, max(accuracies) * 1.2 if accuracies else 1)
        
        plt.tight_layout()
        output_path = self.output_dir / "question_type_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        # Print statistics and tables
        self.print_overall_performance()
        self.create_merged_scene_table()
        self.create_question_type_table()
        
        # Generate plots
        print("Generating plots...")
        self.plot_config_comparison()
        self.plot_question_type_accuracy()
        self.plot_object_tags_top10()
        self.plot_camera_tags_top10()
        
        print("\n" + "="*80)
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*80 + "\n")


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graphs.py <path_to_results.csv> [baseline_results.csv]")
        print("\nExample: python graphs.py evaluation_results/mqa_results_test_20231215_143022.csv")
        print("With baseline: python graphs.py evaluation_results/mqa_results_test.csv evaluation_results/baseline_gpt4o_results.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\nLoading results from: {csv_path}")
    if baseline_path:
        print(f"Loading baseline from: {baseline_path}")
    
    visualizer = MQAResultsVisualizer(csv_path, baseline_path)
    visualizer.generate_all_visualizations()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()