import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_experiment_results(results_dir):
    """Load all experiment results from JSON files."""
    results_dir = Path(results_dir)
    all_results = []
    
    for json_file in sorted(results_dir.glob("*_results.json")):
        with open(json_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)
    
    return all_results


def create_comparison_plots(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors for each optimizer
    colors = {
        'adam': ['#1f77b4', '#4393c3', '#92c5de'],
        'rmsprop': ['#d95f02', '#fc8d59', '#fdb863'],
        'sgd_momentum': ['#2ca02c', '#66c2a4', '#b2e2e2']
    }
    
    # Extract learning rates from optimizer names
    def get_optimizer_lr(opt_name):
        parts = opt_name.split('_lr')
        optimizer = parts[0]
        lr = float(parts[1])
        return optimizer, lr
    
    # Organize results by optimizer
    organized_results = {}
    for result in all_results:
        opt_name = result['optimizer']
        optimizer, lr = get_optimizer_lr(opt_name)
        
        if optimizer not in organized_results:
            organized_results[optimizer] = {}
        organized_results[optimizer][lr] = result
    
    # ========== Plot 1: All Training Losses ==========
    plt.figure(figsize=(14, 6))
    
    for optimizer in ['adam', 'rmsprop', 'sgd_momentum']:
        if optimizer not in organized_results:
            continue
        
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            train_losses = result['train_losses']
            steps = [s for _, s in train_losses]
            losses = [l for l, _ in train_losses]
            
            label = f"{optimizer.upper()} (lr={lr})"
            plt.plot(steps, losses, label=label, color=colors[optimizer][i], alpha=0.7)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison: All Optimizers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== Plot 2: All Validation Losses ==========
    plt.figure(figsize=(14, 6))
    
    for optimizer in ['adam', 'rmsprop', 'sgd_momentum']:
        if optimizer not in organized_results:
            continue
        
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            eval_losses = result['eval_losses']
            steps = [s for _, s in eval_losses]
            losses = [l for l, _ in eval_losses]
            
            label = f"{optimizer.upper()} (lr={lr})"
            plt.plot(steps, losses, marker='o', label=label, 
                    color=colors[optimizer][i], linewidth=2, markersize=6)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Comparison: All Optimizers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_validation_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== Plot 3: All Validation Perplexities ==========
    plt.figure(figsize=(14, 6))
    
    for optimizer in ['adam', 'rmsprop', 'sgd_momentum']:
        if optimizer not in organized_results:
            continue
        
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            eval_ppls = result['eval_perplexities']
            steps = [s for _, s in eval_ppls]
            ppls = [p for p, _ in eval_ppls]
            
            label = f"{optimizer.upper()} (lr={lr})"
            plt.plot(steps, ppls, marker='s', label=label, 
                    color=colors[optimizer][i], linewidth=2, markersize=6)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Perplexity', fontsize=12)
    plt.title('Validation Perplexity Comparison: All Optimizers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_validation_perplexities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== Plot 4: Individual Optimizer Comparisons ==========
    for optimizer in ['adam', 'rmsprop', 'sgd_momentum']:
        if optimizer not in organized_results:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 4a: Training Loss
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            train_losses = result['train_losses']
            steps = [s for _, s in train_losses]
            losses = [l for l, _ in train_losses]
            axes[0].plot(steps, losses, label=f'lr={lr}', 
                        color=colors[optimizer][i], linewidth=2)
        
        axes[0].set_xlabel('Training Steps', fontsize=11)
        axes[0].set_ylabel('Training Loss', fontsize=11)
        axes[0].set_title(f'{optimizer.upper()}: Training Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 4b: Validation Loss
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            eval_losses = result['eval_losses']
            steps = [s for _, s in eval_losses]
            losses = [l for l, _ in eval_losses]
            axes[1].plot(steps, losses, marker='o', label=f'lr={lr}', 
                        color=colors[optimizer][i], linewidth=2, markersize=5)
        
        axes[1].set_xlabel('Training Steps', fontsize=11)
        axes[1].set_ylabel('Validation Loss', fontsize=11)
        axes[1].set_title(f'{optimizer.upper()}: Validation Loss', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 4c: Validation Perplexity
        for i, (lr, result) in enumerate(sorted(organized_results[optimizer].items())):
            eval_ppls = result['eval_perplexities']
            steps = [s for _, s in eval_ppls]
            ppls = [p for p, _ in eval_ppls]
            axes[2].plot(steps, ppls, marker='s', label=f'lr={lr}', 
                        color=colors[optimizer][i], linewidth=2, markersize=5)
        
        axes[2].set_xlabel('Training Steps', fontsize=11)
        axes[2].set_ylabel('Validation Perplexity', fontsize=11)
        axes[2].set_title(f'{optimizer.upper()}: Validation Perplexity', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{optimizer}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All plots saved to {output_dir}")


def create_results_table(all_results, output_dir):
    """
    Create a summary table with final validation loss and perplexity
    for each experiment.
    """
    output_dir = Path(output_dir)
    
    table_data = []
    
    for result in all_results:
        opt_name = result['optimizer']
        parts = opt_name.split('_lr')
        optimizer = parts[0]
        lr = float(parts[1])
        
        # Get final validation metrics
        final_val_loss = result['eval_losses'][-1][0]
        final_val_ppl = result['eval_perplexities'][-1][0]
        
        table_data.append({
            'Optimizer': optimizer.upper(),
            'Learning Rate': lr,
            'Final Val Loss': f"{final_val_loss:.4f}",
            'Final Val Perplexity': f"{final_val_ppl:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    df = df.sort_values(['Optimizer', 'Learning Rate'])
    
    # Save as CSV
    csv_file = output_dir / 'results_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nResults table saved to {csv_file}")
    
    # Print table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Also save as LaTeX for report
    latex_file = output_dir / 'results_summary.tex'
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"LaTeX table saved to {latex_file}")
    
    return df


def find_best_configurations(all_results):
    """Find the best learning rate for each optimizer based on final validation loss."""
    
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS (Based on Final Validation Loss)")
    print("="*80)
    
    # Organize by optimizer
    by_optimizer = {}
    for result in all_results:
        opt_name = result['optimizer']
        parts = opt_name.split('_lr')
        optimizer = parts[0]
        lr = float(parts[1])
        
        if optimizer not in by_optimizer:
            by_optimizer[optimizer] = []
        
        final_val_loss = result['eval_losses'][-1][0]
        final_val_ppl = result['eval_perplexities'][-1][0]
        
        by_optimizer[optimizer].append({
            'lr': lr,
            'val_loss': final_val_loss,
            'val_ppl': final_val_ppl
        })
    
    # Find best for each
    for optimizer in sorted(by_optimizer.keys()):
        configs = by_optimizer[optimizer]
        best_config = min(configs, key=lambda x: x['val_loss'])
        
        print(f"\n{optimizer.upper()}:")
        print(f"  Best Learning Rate: {best_config['lr']}")
        print(f"  Final Validation Loss: {best_config['val_loss']:.4f}")
        print(f"  Final Validation Perplexity: {best_config['val_ppl']:.2f}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to generate all plots and tables."""
    
    results_dir = "./experiments"
    plots_dir = "./plots"
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Results directory: {os.path.abspath(results_dir)}")
    print(f"Plots directory: {os.path.abspath(plots_dir)}")
    print()
    
    print("Loading experiment results...")
    all_results = load_experiment_results(results_dir)
    
    if len(all_results) == 0:
        print(f"No results found in {results_dir}")
        print("Please run the training experiments first!")
        return
    
    print(f"Loaded {len(all_results)} experiment results")
    
    print("\nCreating plots...")
    create_comparison_plots(all_results, plots_dir)
    
    print("\nCreating results table...")
    create_results_table(all_results, plots_dir)
    
    print("\nFinding best configurations...")
    find_best_configurations(all_results)
    
    print("\n✓ Analysis complete!")
    print(f"All outputs saved to: {plots_dir}")


if __name__ == "__main__":
    main()