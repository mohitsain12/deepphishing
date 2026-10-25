"""
MODULE 4: Dataset Understanding & Exploratory Data Analysis (EDA)
Performs comprehensive analysis and visualization of the phishing dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhishingDatasetExplorer:
    """
    Comprehensive EDA for the phishing URL dataset.
    Generates statistics, visualizations, and insights.
    """
    
    def __init__(self, data_path: str = "data/raw/phishing_dataset.csv",
                 figures_dir: str = "reports/figures"):
        """
        Initialize the explorer.
        
        Args:
            data_path: Path to the raw dataset
            figures_dir: Directory to save figures
        """
        self.data_path = data_path
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset: {self.df.shape}")
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def basic_statistics(self) -> dict:
        """
        Compute and display basic dataset statistics.
        
        Returns:
            Dictionary with basic statistics
        """
        logger.info("Computing basic statistics...")
        
        stats = {
            'total_samples': len(self.df),
            'features': list(self.df.columns),
            'num_features': len(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
        }
        
        if 'status' in self.df.columns:
            stats['class_distribution'] = self.df['status'].value_counts().to_dict()
            stats['class_balance_ratio'] = self.df['status'].value_counts().min() / self.df['status'].value_counts().max()
        
        # Print statistics
        print("\n" + "=" * 60)
        print("DATASET BASIC STATISTICS")
        print("=" * 60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of features: {stats['num_features']}")
        print(f"Duplicate rows: {stats['duplicates']}")
        print(f"\nMissing Values:")
        for col, count in stats['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count} ({stats['missing_percentage'][col]:.2f}%)")
        if not any(v > 0 for v in stats['missing_values'].values()):
            print("  No missing values found!")
        
        if 'class_distribution' in stats:
            print(f"\nClass Distribution:")
            for cls, count in stats['class_distribution'].items():
                label = "Phishing" if cls == 1 else "Legitimate"
                print(f"  {label} ({cls}): {count} ({count/stats['total_samples']*100:.1f}%)")
            print(f"  Balance Ratio: {stats['class_balance_ratio']:.4f}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        print(f"\nData Types:")
        print(self.df.dtypes)
        print(f"\nDescriptive Statistics:")
        print(self.df.describe())
        
        return stats
    
    def analyze_url_patterns(self):
        """Analyze URL-specific patterns in the dataset."""
        logger.info("Analyzing URL patterns...")
        
        if 'url' not in self.df.columns:
            logger.warning("No 'url' column found, skipping URL analysis")
            return
        
        # URL length analysis
        self.df['_url_length'] = self.df['url'].str.len()
        
        print("\n" + "=" * 60)
        print("URL PATTERN ANALYSIS")
        print("=" * 60)
        
        # By class
        if 'status' in self.df.columns:
            for status in self.df['status'].unique():
                label = "Phishing" if status == 1 else "Legitimate"
                subset = self.df[self.df['status'] == status]
                print(f"\n{label} URLs:")
                print(f"  Avg length: {subset['_url_length'].mean():.1f}")
                print(f"  Min length: {subset['_url_length'].min()}")
                print(f"  Max length: {subset['_url_length'].max()}")
                print(f"  Median length: {subset['_url_length'].median():.1f}")
        
        # Protocol analysis
        self.df['_protocol'] = self.df['url'].apply(
            lambda x: 'https' if str(x).startswith('https') else 'http'
        )
        print(f"\nProtocol Distribution:")
        print(self.df['_protocol'].value_counts())
        
        # Clean up temporary columns
        self.df.drop(columns=[c for c in self.df.columns if c.startswith('_')], inplace=True)
    
    def plot_class_distribution(self):
        """Plot the distribution of phishing vs legitimate URLs."""
        if 'status' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        counts = self.df['status'].value_counts()
        labels = ['Legitimate', 'Phishing']
        colors = ['#2ecc71', '#e74c3c']
        
        axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5)
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        for i, (label, count) in enumerate(zip(labels, counts.values)):
            axes[0].text(i, count + 50, str(count), ha='center', fontsize=12, fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts.values, labels=labels, colors=colors,
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'class_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved class_distribution.png")
    
    def plot_url_length_distribution(self):
        """Plot URL length distribution for each class."""
        if 'url' not in self.df.columns:
            return
        
        self.df['url_length'] = self.df['url'].str.len()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if 'status' in self.df.columns:
            # Histogram by class
            for status, color, label in [(0, '#2ecc71', 'Legitimate'), (1, '#e74c3c', 'Phishing')]:
                subset = self.df[self.df['status'] == status]
                axes[0].hist(subset['url_length'], bins=50, alpha=0.6,
                           label=label, color=color, edgecolor='white')
            axes[0].set_title('URL Length Distribution by Class', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('URL Length', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].legend(fontsize=11)
            
            # Box plot by class
            data_to_plot = [
                self.df[self.df['status'] == 0]['url_length'],
                self.df[self.df['status'] == 1]['url_length']
            ]
            try:
                bp = axes[1].boxplot(data_to_plot, tick_labels=['Legitimate', 'Phishing'],
                                    patch_artist=True)
            except TypeError:
                bp = axes[1].boxplot(data_to_plot, labels=['Legitimate', 'Phishing'],
                                    patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            axes[1].set_title('URL Length Box Plot by Class', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('URL Length', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'url_length_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self.df.drop(columns=['url_length'], inplace=True, errors='ignore')
        logger.info("Saved url_length_distribution.png")
    
    def plot_feature_correlations(self):
        """Plot correlation heatmap for numerical features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.info("Not enough numerical features for correlation plot")
            return
        
        corr = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True if len(numeric_cols) <= 15 else False,
                    fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved correlation_heatmap.png")
    
    def plot_special_char_analysis(self):
        """Analyze special character frequency in URLs."""
        if 'url' not in self.df.columns:
            return
        
        special_chars = ['.', '-', '_', '/', '?', '=', '@', '&', '!', '~', '#', '%']
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, char in enumerate(special_chars):
            self.df[f'_count_{char}'] = self.df['url'].str.count(f'\\{char}' if char in '.?*+[](){}|^$\\' else char)
            
            if 'status' in self.df.columns:
                legit = self.df[self.df['status'] == 0][f'_count_{char}']
                phish = self.df[self.df['status'] == 1][f'_count_{char}']
                
                axes[i].hist(legit, bins=20, alpha=0.6, label='Legitimate', color='#2ecc71')
                axes[i].hist(phish, bins=20, alpha=0.6, label='Phishing', color='#e74c3c')
                axes[i].set_title(f"'{char}' Count", fontsize=11, fontweight='bold')
                axes[i].legend(fontsize=8)
            
            self.df.drop(columns=[f'_count_{char}'], inplace=True)
        
        plt.suptitle('Special Character Frequency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'special_char_analysis.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved special_char_analysis.png")
    
    def generate_eda_summary(self) -> str:
        """
        Generate a comprehensive EDA summary text.
        
        Returns:
            Summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("EXPLORATORY DATA ANALYSIS SUMMARY")
        summary.append("=" * 60)
        summary.append(f"\nDataset: {self.data_path}")
        summary.append(f"Total Samples: {len(self.df)}")
        summary.append(f"Features: {len(self.df.columns)}")
        summary.append(f"Missing Values: {self.df.isnull().sum().sum()}")
        summary.append(f"Duplicates: {self.df.duplicated().sum()}")
        
        if 'status' in self.df.columns:
            counts = self.df['status'].value_counts()
            summary.append(f"\nClass 0 (Legitimate): {counts.get(0, 0)}")
            summary.append(f"Class 1 (Phishing): {counts.get(1, 0)}")
            ratio = counts.min() / counts.max()
            summary.append(f"Imbalance Ratio: {ratio:.4f}")
            
            if ratio > 0.8:
                summary.append("→ Dataset is well-balanced ✓")
            elif ratio > 0.5:
                summary.append("→ Slight class imbalance detected")
            else:
                summary.append("→ Significant class imbalance! Consider oversampling/undersampling")
        
        summary_text = "\n".join(summary)
        
        # Save summary
        summary_path = os.path.join(self.figures_dir, '..', 'eda_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        return summary_text
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        logger.info("=" * 60)
        logger.info("MODULE 4: Dataset Understanding & EDA")
        logger.info("=" * 60)
        
        # Basic statistics
        stats = self.basic_statistics()
        
        # URL pattern analysis
        self.analyze_url_patterns()
        
        # Generate plots
        self.plot_class_distribution()
        self.plot_url_length_distribution()
        self.plot_feature_correlations()
        self.plot_special_char_analysis()
        
        # Generate summary
        summary = self.generate_eda_summary()
        print(summary)
        
        logger.info("EDA complete! Check reports/figures/ for visualizations.")
        return stats


def main():
    """Main function to run EDA."""
    data_path = "data/raw/phishing_dataset.csv"
    
    if not os.path.exists(data_path):
        print("Dataset not found. Running data acquisition first...")
        from src.data.download_dataset import PhishingDatasetDownloader
        downloader = PhishingDatasetDownloader()
        downloader.prepare_dataset()
    
    explorer = PhishingDatasetExplorer(data_path)
    explorer.run_full_eda()


if __name__ == "__main__":
    main()
