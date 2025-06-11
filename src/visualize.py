import matplotlib.pyplot as plt

def plot_visualizer(history, title="Model Training History"):
    """
    Wizualizuje historię trenowania modelu multi-head.
    Dostosowane do specyficznych nazw metryk dla każdego head'a.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title} - Training History', fontsize=16)
    
    # Subplot 1: Total Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Specific losses based on dataset
    if 'score_output_loss' in history.history:  # ASAP
        axes[0, 1].plot(history.history['score_output_loss'], label='Training Score Loss', color='green')
        axes[0, 1].plot(history.history['val_score_output_loss'], label='Validation Score Loss', color='orange')
        axes[0, 1].set_title('ASAP Score Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
    elif 'readability_output_loss' in history.history:  # CommonLit
        axes[0, 1].plot(history.history['readability_output_loss'], label='Training Readability Loss', color='green')
        axes[0, 1].plot(history.history['val_readability_output_loss'], label='Validation Readability Loss', color='orange')
        axes[0, 1].set_title('CommonLit Readability Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
    elif 'jfleg_output_loss' in history.history:  # JFLEG
        axes[0, 1].plot(history.history['jfleg_output_loss'], label='Training JFLEG Loss', color='green')
        axes[0, 1].plot(history.history['val_jfleg_output_loss'], label='Validation JFLEG Loss', color='orange')
        axes[0, 1].set_title('JFLEG Binary Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: MAE dla regresji lub Accuracy dla klasyfikacji
    if 'score_output_mae' in history.history:  # ASAP MAE
        axes[1, 0].plot(history.history['score_output_mae'], label='Training MAE', color='purple')
        axes[1, 0].plot(history.history['val_score_output_mae'], label='Validation MAE', color='brown')
        axes[1, 0].set_title('ASAP Mean Absolute Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    elif 'readability_output_mae' in history.history:  # CommonLit MAE
        axes[1, 0].plot(history.history['readability_output_mae'], label='Training MAE', color='purple')
        axes[1, 0].plot(history.history['val_readability_output_mae'], label='Validation MAE', color='brown')
        axes[1, 0].set_title('CommonLit Mean Absolute Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    elif 'jfleg_output_accuracy' in history.history:  # JFLEG Accuracy
        axes[1, 0].plot(history.history['jfleg_output_accuracy'], label='Training Accuracy', color='purple')
        axes[1, 0].plot(history.history['val_jfleg_output_accuracy'], label='Validation Accuracy', color='brown')
        axes[1, 0].set_title('JFLEG Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Learning Rate (jeśli dostępne)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', color='black')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')  # Log scale dla lepszej wizualizacji
    else:
        # Jeśli nie ma lr, pokaż wszystkie dostępne metryki
        available_metrics = [key for key in history.history.keys() 
                           if not key.startswith('val_') and key != 'loss']
        if available_metrics:
            metric_name = available_metrics[0]
            axes[1, 1].plot(history.history[metric_name], label=f'Training {metric_name}', color='teal')
            if f'val_{metric_name}' in history.history:
                axes[1, 1].plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name}', color='coral')
            axes[1, 1].set_title(f'Additional Metric: {metric_name}')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel(metric_name)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No additional\nmetrics available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Additional Metrics')
    
    plt.tight_layout()
    plt.show()

def plot_mse(mse_values, datasets):
    """
    Wizualizuje MSE dla różnych datasetów.
    """
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    bars = plt.bar(datasets, mse_values, color=colors[:len(datasets)])
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Mean Squared Error on Test Sets', fontsize=14, fontweight='bold')
    
    # Dodaj wartości na słupkach
    for i, (bar, v) in enumerate(zip(bars, mse_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01, 
                f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Dodaj różne kolory dla lepszej wizualizacji
    for i, bar in enumerate(bars):
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    plt.show()

def plot_all_metrics_comparison(histories, titles):
    """
    Porównuje metryki z różnych przebiegów trenowania.
    histories: lista obiektów history
    titles: lista tytułów odpowiadających każdemu history
    """
    if len(histories) != len(titles):
        raise ValueError("Liczba histories musi być równa liczbie titles")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Porównanie wszystkich treningów', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Loss comparison
    axes[0, 0].set_title('Total Loss Comparison')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        axes[0, 0].plot(history.history['loss'], label=f'{title} Train', color=color, linestyle='-')
        axes[0, 0].plot(history.history['val_loss'], label=f'{title} Val', color=color, linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Specific losses
    axes[0, 1].set_title('Specific Task Losses')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        # Znajdź specyficzny loss dla tego datasetu
        for key in history.history.keys():
            if 'output_loss' in key and not key.startswith('val_'):
                axes[0, 1].plot(history.history[key], label=f'{title} {key}', color=color)
                break
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE/Accuracy comparison
    axes[1, 0].set_title('Task-specific Metrics')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        # Znajdź MAE lub accuracy
        for key in history.history.keys():
            if ('mae' in key or 'accuracy' in key) and not key.startswith('val_'):
                axes[1, 0].plot(history.history[key], label=f'{title} {key}', color=color)
                break
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Metric Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation metrics
    axes[1, 1].set_title('Validation Metrics')
    for i, (history, title) in enumerate(zip(histories, titles)):
        color = colors[i % len(colors)]
        for key in history.history.keys():
            if key.startswith('val_') and ('mae' in key or 'accuracy' in key):
                axes[1, 1].plot(history.history[key], label=f'{title} {key}', color=color)
                break
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Metric')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()