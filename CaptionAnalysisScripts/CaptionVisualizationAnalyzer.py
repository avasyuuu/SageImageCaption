import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CaptionVisualizationAnalyzer:
    def __init__(self, evaluation_json_path):
        """Initialize with evaluation results JSON"""
        with open(evaluation_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.models = list(self.data['model_performance'].keys())
        self.output_dir = None
        
        self.model_colors = {
            'BLIP': '#1f77b4',
            'Florence-2': '#ff7f0e', 
            'LLaVA': '#2ca02c',
            'Moondream': '#d62728'
        }
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_output_directory(self, base_path="visualizations"):
        """Create output directory for visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{base_path}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/model_comparison", exist_ok=True)
        os.makedirs(f"{self.output_dir}/statistical_analysis", exist_ok=True)
        print(f"📁 Created visualization directory: {self.output_dir}")
        return self.output_dir
    
    def extract_all_scores(self):
        """Extract all scores into a structured DataFrame"""
        data_rows = []
        
        for img_eval in self.data['image_evaluations']:
            filename = Path(img_eval['filename']).name
            
            for model_name, eval_data in img_eval['model_evaluations'].items():
                if 'accuracy_score' in eval_data and eval_data['accuracy_score'] >= 0:
                    row = {
                        'image': filename,
                        'model': model_name,
                        'accuracy': eval_data.get('accuracy_score', 0),
                        'object_identification': eval_data.get('object_identification_score', 0),
                        'scene_understanding': eval_data.get('scene_understanding_score', 0),
                        'missing_elements_count': len(eval_data.get('missing_elements', [])),
                        'incorrect_elements_count': len(eval_data.get('incorrect_elements', []))
                    }
                    data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def create_model_comparison_boxplots(self, df):
        """Create box plots comparing score distributions across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Score Distributions', fontsize=16)
        
        df['overall'] = (
            df['accuracy'] * 0.4 + 
            df['object_identification'] * 0.3 + 
            df['scene_understanding'] * 0.3
        )
        
        attributes = [
            ('accuracy', 'Accuracy Score'),
            ('object_identification', 'Object Identification Score'),
            ('scene_understanding', 'Scene Understanding Score'),
            ('overall', 'Overall Score (Weighted)')
        ]
        
        for idx, (attr, title) in enumerate(attributes):
            ax = axes[idx // 2, idx % 2]
            
            box_data = [df[df['model'] == model][attr].values for model in self.models]
            bp = ax.boxplot(box_data, labels=self.models, patch_artist=True)
            
            for patch, model in zip(bp['boxes'], self.models):
                patch.set_facecolor(self.model_colors.get(model, '#888888'))
                patch.set_alpha(0.7)
            
            means = [np.mean(data) for data in box_data]
            ax.scatter(range(1, len(self.models) + 1), means, 
                      color='red', s=100, zorder=3, label='Mean')
            
            ax.set_title(title)
            ax.set_ylabel('Score')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison/boxplots_all_attributes.png", dpi=150)
        plt.close()
    
    def create_score_distribution_histograms(self, df):
        """Create histograms showing score distributions for each model"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Score Distribution by Model', fontsize=16)
        
        for idx, model in enumerate(self.models):
            ax = axes[idx // 2, idx % 2]
            model_data = df[df['model'] == model]
            
            attributes = ['accuracy', 'object_identification', 'scene_understanding']
            colors = ['blue', 'green', 'orange']
            
            for attr, color in zip(attributes, colors):
                ax.hist(model_data[attr], bins=20, alpha=0.5, label=attr.replace('_', ' ').title(), 
                       color=color, edgecolor='black')
            
            ax.set_title(f'{model} Score Distribution')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison/score_distributions.png", dpi=150)
        plt.close()
    
    def calculate_advanced_statistics(self, df):
        """Calculate advanced statistics for model comparison"""
        stats_data = {}
        
        for model in self.models:
            model_data = df[df['model'] == model]
            stats_data[model] = {}
            
            for attr in ['accuracy', 'object_identification', 'scene_understanding']:
                scores = model_data[attr].values
                
                stats_data[model][attr] = {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                    'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                    'skewness': stats.skew(scores),
                    'kurtosis': stats.kurtosis(scores),
                    'percentile_10': np.percentile(scores, 10),
                    'percentile_90': np.percentile(scores, 90),
                    'consistency_score': 100 - (np.std(scores) / np.mean(scores) * 100) if np.mean(scores) > 0 else 0
                }
        
        return stats_data
    
    def create_statistical_summary_plots(self, df, stats_data):
        """Create comprehensive statistical summary visualizations"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model in self.models:
            avg_performance = np.mean([
                stats_data[model]['accuracy']['mean'],
                stats_data[model]['object_identification']['mean'],
                stats_data[model]['scene_understanding']['mean']
            ])
            
            avg_consistency = np.mean([
                stats_data[model]['accuracy']['consistency_score'],
                stats_data[model]['object_identification']['consistency_score'],
                stats_data[model]['scene_understanding']['consistency_score']
            ])
            
            color = self.model_colors.get(model, '#888888')
            ax.scatter(avg_consistency, avg_performance, s=200, color=color, 
                      label=model, alpha=0.7, edgecolors='black', linewidth=2)
            
            ax.annotate(model, (avg_consistency, avg_performance), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Consistency Score (100 - CV%)', fontsize=12)
        ax.set_ylabel('Average Performance Score', fontsize=12)
        ax.set_title('Model Performance vs Consistency', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
        
        ax.text(85, 85, 'High Performance\nHigh Consistency', ha='center', va='center', 
                alpha=0.5, fontsize=10, style='italic')
        ax.text(85, 55, 'Low Performance\nHigh Consistency', ha='center', va='center', 
                alpha=0.5, fontsize=10, style='italic')
        ax.text(55, 85, 'High Performance\nLow Consistency', ha='center', va='center', 
                alpha=0.5, fontsize=10, style='italic')
        ax.text(55, 55, 'Low Performance\nLow Consistency', ha='center', va='center', 
                alpha=0.5, fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/statistical_analysis/performance_vs_consistency.png", dpi=150)
        plt.close()
    
    def create_heatmap_comparison(self, df):
        """Create heatmap showing all scores for all models and images"""
        pivot_data = df.pivot_table(
            index='image', 
            columns='model', 
            values='accuracy', 
            aggfunc='first'
        )
        
        plt.figure(figsize=(12, max(8, len(pivot_data) * 0.3)))
        
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Accuracy Score'})
        
        plt.title('Accuracy Scores Heatmap: Models vs Images', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Image', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison/accuracy_heatmap.png", dpi=150)
        plt.close()
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("\n🎨 Generating visualizations...")
        
        if not self.output_dir:
            self.create_output_directory()
        
        df = self.extract_all_scores()
        
        if df.empty:
            print("❌ No valid evaluation data found!")
            return
        
        print(f"📊 Processing {len(df)} evaluation records from {len(df['image'].unique())} images")
        
        print("  • Creating model comparison box plots...")
        self.create_model_comparison_boxplots(df)
        
        print("  • Creating score distribution histograms...")
        self.create_score_distribution_histograms(df)
        
        print("  • Creating accuracy heatmap...")
        self.create_heatmap_comparison(df)
        
        print("  • Calculating advanced statistics...")
        stats_data = self.calculate_advanced_statistics(df)
        
        print("  • Creating statistical summary plots...")
        self.create_statistical_summary_plots(df, stats_data)
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}/")
        
        return self.output_dir
    