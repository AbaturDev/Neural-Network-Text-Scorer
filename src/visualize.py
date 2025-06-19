import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_training_history(history, title="Multi-Head MLP Training History", save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title}', fontsize=18, fontweight='bold')
    
    colors = {
        'train': '#2E86AB',
        'val': '#A23B72',
        'asap': '#F18F01',
        'commonlit': '#C73E1D',
        'jfleg': '#4CAF50'
    }
    
    axes[0, 0].plot(history.history['loss'], label='Training Loss', 
                    color=colors['train'], linewidth=2.5)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', 
                    color=colors['val'], linewidth=2.5)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    if 'score_output_loss' in history.history:
        axes[0, 1].plot(history.history['score_output_loss'], 
                       label='Training ASAP Loss', color=colors['asap'], linewidth=2)
        axes[0, 1].plot(history.history['val_score_output_loss'], 
                       label='Validation ASAP Loss', color=colors['asap'], 
                       linestyle='--', linewidth=2)
        axes[0, 1].set_title('ASAP Score Loss (MSE)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MSE Loss', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'readability_output_loss' in history.history:
        axes[0, 2].plot(history.history['readability_output_loss'], 
                       label='Training CommonLit Loss', color=colors['commonlit'], linewidth=2)
        axes[0, 2].plot(history.history['val_readability_output_loss'], 
                       label='Validation CommonLit Loss', color=colors['commonlit'], 
                       linestyle='--', linewidth=2)
        axes[0, 2].set_title('CommonLit Readability Loss (MSE)', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('MSE Loss', fontsize=12)
        axes[0, 2].legend(fontsize=11)
        axes[0, 2].grid(True, alpha=0.3)
    
    if 'jfleg_output_loss' in history.history:
        axes[1, 0].plot(history.history['jfleg_output_loss'], 
                       label='Training JFLEG Loss', color=colors['jfleg'], linewidth=2)
        axes[1, 0].plot(history.history['val_jfleg_output_loss'], 
                       label='Validation JFLEG Loss', color=colors['jfleg'], 
                       linestyle='--', linewidth=2)
        axes[1, 0].set_title('JFLEG Error Detection Loss (BCE)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Binary Crossentropy', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'score_output_mae' in history.history and 'readability_output_mae' in history.history:
        ax_mae = axes[1, 1]
        ax_acc = ax_mae.twinx()
        
        line1 = ax_mae.plot(history.history['score_output_mae'], 
                           label='ASAP MAE', color=colors['asap'], linewidth=2)
        line2 = ax_mae.plot(history.history['val_score_output_mae'], 
                           label='ASAP MAE (Val)', color=colors['asap'], 
                           linestyle='--', linewidth=2)
        line3 = ax_mae.plot(history.history['readability_output_mae'], 
                           label='CommonLit MAE', color=colors['commonlit'], linewidth=2)
        line4 = ax_mae.plot(history.history['val_readability_output_mae'], 
                           label='CommonLit MAE (Val)', color=colors['commonlit'], 
                           linestyle='--', linewidth=2)
        
        if 'jfleg_output_accuracy' in history.history:
            line5 = ax_acc.plot(history.history['jfleg_output_accuracy'], 
                               label='JFLEG Accuracy', color=colors['jfleg'], linewidth=2)
            line6 = ax_acc.plot(history.history['val_jfleg_output_accuracy'], 
                               label='JFLEG Accuracy (Val)', color=colors['jfleg'], 
                               linestyle='--', linewidth=2)
            ax_acc.set_ylabel('Accuracy', fontsize=12, color=colors['jfleg'])
            ax_acc.tick_params(axis='y', labelcolor=colors['jfleg'])
        
        ax_mae.set_title('MAE (Regression) & Accuracy (Classification)', fontsize=14, fontweight='bold')
        ax_mae.set_xlabel('Epoch', fontsize=12)
        ax_mae.set_ylabel('Mean Absolute Error', fontsize=12)
        
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        if 'jfleg_output_accuracy' in history.history:
            lines += line5 + line6
            labels += [l.get_label() for l in line5 + line6]
        ax_mae.legend(lines, labels, loc='upper right', fontsize=10)
        ax_mae.grid(True, alpha=0.3)
    
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'], label='Learning Rate', 
                       color='black', linewidth=2.5)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch', fontsize=12)
        axes[1, 2].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend(fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        if 'jfleg_output_precision' in history.history:
            axes[1, 2].plot(history.history['jfleg_output_precision'], 
                           label='JFLEG Precision', color='blue', linewidth=2)
            axes[1, 2].plot(history.history['val_jfleg_output_precision'], 
                           label='JFLEG Precision (Val)', color='blue', 
                           linestyle='--', linewidth=2)
            if 'jfleg_output_recall' in history.history:
                axes[1, 2].plot(history.history['jfleg_output_recall'], 
                               label='JFLEG Recall', color='red', linewidth=2)
                axes[1, 2].plot(history.history['val_jfleg_output_recall'], 
                               label='JFLEG Recall (Val)', color='red', 
                               linestyle='--', linewidth=2)
            axes[1, 2].set_title('JFLEG Precision & Recall', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch', fontsize=12)
            axes[1, 2].set_ylabel('Score', fontsize=12)
            axes[1, 2].legend(fontsize=11)
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_evaluation_results(evaluation_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results', fontsize=18, fontweight='bold')
    
    regression_tasks = []
    mse_values = []
    mae_values = []
    
    if 'asap' in evaluation_results:
        regression_tasks.append('ASAP\nScoring')
        mse_values.append(evaluation_results['asap']['mse'])
        mae_values.append(evaluation_results['asap']['mae'])
    
    if 'commonlit' in evaluation_results:
        regression_tasks.append('CommonLit\nReadability')
        mse_values.append(evaluation_results['commonlit']['mse'])
        mae_values.append(evaluation_results['commonlit']['mae'])
    
    if regression_tasks:
        x = np.arange(len(regression_tasks))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, mse_values, width, label='MSE', 
                              color='#FF6B6B', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, mae_values, width, label='MAE', 
                              color='#4ECDC4', alpha=0.8)
        
        axes[0, 0].set_title('Regression Tasks Error Metrics', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Error Value', fontsize=12)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(regression_tasks)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].annotate(f'{height:.4f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontweight='bold')
    
    if 'jfleg' in evaluation_results:
        cm = evaluation_results['jfleg']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Errors', 'Has Errors'],
                   yticklabels=['No Errors', 'Has Errors'],
                   ax=axes[0, 1])
        axes[0, 1].set_title('JFLEG Error Detection\nConfusion Matrix', 
                           fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted', fontsize=12)
        axes[0, 1].set_ylabel('Actual', fontsize=12)
    
    datasets = []
    sample_counts = []
    
    for dataset, results in evaluation_results.items():
        if 'n_samples' in results:
            datasets.append(dataset.upper())
            sample_counts.append(results['n_samples'])
    
    if datasets:
        colors_samples = ['#FF9F43', '#6C5CE7', '#00B894']
        bars = axes[1, 0].bar(datasets, sample_counts, 
                             color=colors_samples[:len(datasets)], alpha=0.8)
        axes[1, 0].set_title('Test Set Sample Sizes', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Samples', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].annotate(f'{int(height)}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontweight='bold')
    
    if 'jfleg' in evaluation_results and 'class_distribution' in evaluation_results['jfleg']:
        class_dist = evaluation_results['jfleg']['class_distribution']
        labels = ['No Errors', 'Has Errors']
        sizes = [class_dist['no_errors'], class_dist['has_errors']]
        colors_pie = ['#74B9FF', '#FD79A8']
        
        wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=labels, colors=colors_pie,
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('JFLEG Test Set\nClass Distribution', 
                           fontsize=14, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation results plot saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(histories, titles, evaluation_results_list=None, save_path=None):
    if len(histories) != len(titles):
        raise ValueError("Liczba histories musi być równa liczbie titles")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison Across Experiments', fontsize=18, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    axes[0, 0].set_title('Total Loss Comparison', fontsize=14, fontweight='bold')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        axes[0, 0].plot(history.history['loss'], label=f'{title} (Train)', 
                       color=color, linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label=f'{title} (Val)', 
                       color=color, linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('JFLEG Accuracy Comparison', fontsize=14, fontweight='bold')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        if 'jfleg_output_accuracy' in history.history:
            axes[0, 1].plot(history.history['jfleg_output_accuracy'], 
                           label=f'{title} (Train)', color=color, linewidth=2)
            axes[0, 1].plot(history.history['val_jfleg_output_accuracy'], 
                           label=f'{title} (Val)', color=color, 
                           linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    if evaluation_results_list and len(evaluation_results_list) == len(titles):
        metrics_df = []
        for i, (results, title) in enumerate(zip(evaluation_results_list, titles)):
            row = {'Model': title}
            if 'asap' in results:
                row['ASAP_MSE'] = results['asap']['mse']
                row['ASAP_MAE'] = results['asap']['mae']
            if 'commonlit' in results:
                row['CommonLit_MSE'] = results['commonlit']['mse']
                row['CommonLit_MAE'] = results['commonlit']['mae']
            if 'jfleg' in results:
                row['JFLEG_Accuracy'] = results['jfleg']['accuracy']
            metrics_df.append(row)
        
        metrics_df = pd.DataFrame(metrics_df)
        
        metrics_to_plot = [col for col in metrics_df.columns if col != 'Model']
        if metrics_to_plot:
            x = np.arange(len(titles))
            width = 0.8 / len(metrics_to_plot)
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns:
                    values = metrics_df[metric].fillna(0)
                    bars = axes[1, 0].bar(x + i * width, values, width, 
                                         label=metric, alpha=0.8)
                    
                    for bar, val in zip(bars, values):
                        if val > 0:
                            axes[1, 0].annotate(f'{val:.3f}',
                                              xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                              xytext=(0, 3),
                                              textcoords="offset points",
                                              ha='center', va='bottom', fontsize=8)
            
            axes[1, 0].set_title('Final Performance Metrics', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Metric Value', fontsize=12)
            axes[1, 0].set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
            axes[1, 0].set_xticklabels(titles, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate Schedule Comparison', fontsize=14, fontweight='bold')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label=title, 
                           color=color, linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_loss_weights_analysis(history, loss_weights, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loss Weights Analysis', fontsize=16, fontweight='bold')

    weighted_losses = []
    unweighted_losses = []
    task_names = []
    
    for task, weight in loss_weights.items():
        task_key = f"{task}_loss"
        if task_key in history.history:
            weighted_losses.append(np.array(history.history[task_key]) * weight)
            unweighted_losses.append(history.history[task_key])
            task_names.append(task.replace('_output', '').upper())
    
    if weighted_losses:
        epochs = range(1, len(weighted_losses[0]) + 1)
        
        for i, (weighted, unweighted, name) in enumerate(zip(weighted_losses, unweighted_losses, task_names)):
            axes[0, 0].plot(epochs, weighted, label=f'{name} (Weighted)', linewidth=2)
            axes[0, 1].plot(epochs, unweighted, label=f'{name} (Original)', linewidth=2)
        
        axes[0, 0].set_title('Weighted Task Losses', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Weighted Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Original Task Losses', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Original Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    weight_names = [name.replace('_output', '').upper() for name in loss_weights.keys()]
    weight_values = list(loss_weights.values())
    
    colors_weights = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = axes[1, 0].bar(weight_names, weight_values, color=colors_weights, alpha=0.8)
    axes[1, 0].set_title('Loss Weight Configuration', fontsize=14)
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, weight in zip(bars, weight_values):
        axes[1, 0].annotate(f'{weight}',
                          xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold')
    
    if weighted_losses:
        final_contributions = [losses[-1] for losses in weighted_losses]
        total_final = sum(final_contributions)
        percentages = [(contrib/total_final)*100 for contrib in final_contributions]
        
        wedges, texts, autotexts = axes[1, 1].pie(percentages, labels=task_names, 
                                                 autopct='%1.1f%%', startangle=90,
                                                 colors=colors_weights)
        axes[1, 1].set_title('Final Loss Contribution (%)', fontsize=14)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss weights analysis plot saved to: {save_path}")
    
    plt.show()


def visualize_training(history, title="MLP Multi-Head Model", save_dir=None):
    """
    Główna funkcja do wizualizacji - wrapper dla łatwego użycia
    """
    save_path = None
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_training.png")
    
    plot_training_history(history, title, save_path)


def visualize_evaluation(results, save_dir=None):
    """
    Główna funkcja do wizualizacji wyników ewaluacji
    """
    save_path = None
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "evaluation_results.png")
    
    plot_evaluation_results(results, save_path)


def create_comprehensive_report(history, evaluation_results, loss_weights, 
                               title="MLP Multi-Head Model Report", save_dir=None):
    """
    Tworzy kompletny raport wizualny z trenowania i ewaluacji
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    training_save_path = None
    if save_dir:
        training_save_path = os.path.join(save_dir, f"{title}_training_history.png")
    plot_training_history(history, title, training_save_path)
    
    eval_save_path = None
    if save_dir:
        eval_save_path = os.path.join(save_dir, f"{title}_evaluation.png")
    plot_evaluation_results(evaluation_results, eval_save_path)
    
    weights_save_path = None
    if save_dir:
        weights_save_path = os.path.join(save_dir, f"{title}_loss_weights.png")
    plot_loss_weights_analysis(history, loss_weights, weights_save_path)
