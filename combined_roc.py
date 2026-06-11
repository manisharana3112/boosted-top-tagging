import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. Plotting Configuration
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 2,
    'lines.linewidth': 2.5
})

# ==========================================
# 2. Define Paths and Exact Keys
# ==========================================
BASE_DIR = '/home/sachit/ML/jetclass_Results_1M'

# We now define the exact true/pred keys for each model based on your terminal output
model_folders = {
    'XGBoost (BDT_Max)':   {'folder': 'BDT_Max',     'color': '#9b59b6', 'ls': '-', 'true_key': 'y_test', 'pred_key': 'y_prob_test'},
    'CNN (CNN_Opt)':       {'folder': 'CNN_Opt',     'color': '#3498db', 'ls': '-', 'true_key': 'test_y', 'pred_key': 'test_probs'},
    'ResNeXt (rexnext)':   {'folder': 'rexnext',     'color': '#2ecc71', 'ls': '-', 'true_key': 'test_y', 'pred_key': 'test_probs'},
    'ParticleNet':         {'folder': 'ParticleNet', 'color': '#e74c3c', 'ls': '-', 'true_key': 'test_y', 'pred_key': 'test_probs'},
    'UParT (UParT_1)':     {'folder': 'UParT_1',     'color': '#f39c12', 'ls': '-', 'true_key': 'test_y', 'pred_key': 'test_probs'}
}

# ==========================================
# 3. Generate the ROC Curve Plot
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

for name, info in model_folders.items():
    # Updated to look for 'all_scores.npz'
    file_path = os.path.join(BASE_DIR, info['folder'], 'all_scores.npz')
    
    try:
        data = np.load(file_path)
        
        # Load arrays using the model-specific keys
        y_true = data[info['true_key']]
        y_pred = data[info['pred_key']]
        
        # Calculate FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=info['color'], linestyle=info['ls'], 
                label=f'{name} (AUC = {roc_auc:.4f})')
                
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}. Skipping {name}.")
    except KeyError as e:
        print(f"Error: Key {e} not found in {file_path}. Available keys are: {data.files}")

# Plot the Random Guess baseline
ax.plot([1e-4, 1.0], [1e-4, 1.0], color='navy', linestyle='--', lw=2, label='Random Guess')

# ==========================================
# 4. Axes Formatting & Log Scale
# ==========================================
#ax.set_xscale('log')
ax.set_yscale('log')

# Set limits focusing on the relevant physics region
ax.set_xlim([1e-4, 1.0]) 
ax.set_ylim([0.0, 1.05])

# Grid and Labels
ax.grid(True, which="both", ls=":", alpha=0.7, color='gray')
ax.set_xlabel('False Positive Rate (QCD Mistag)', fontweight='bold')
ax.set_ylabel('True Positive Rate (Top Efficiency)', fontweight='bold')
ax.set_title('ROC Curve ', fontweight='bold', pad=15)

# Legend positioning
ax.legend(loc="lower right", framealpha=1.0, edgecolor='black', fancybox=True)

# Save the figure
output_path = os.path.join(BASE_DIR, 'combined_roc_xy_log.png')
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"Plot saved successfully to: {output_path}")

# plt.show() # Disabled to prevent WSL non-interactive warnings
