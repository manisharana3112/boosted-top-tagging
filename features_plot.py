import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 1. DIRECTORY SETTINGS
# ==========================================
INPUT_H5 = "/home/sachit/ML/processed_data/jetclass_1M/BDT_optimized_dataset.h5"
RESULTS_DIR = "/home/sachit/ML/jetclass_Results_1M/BDT_Max/feature_plots"
# ==========================================

def set_plot_style():
	"""Configures matplotlib ."""
	sns.set_theme(style="whitegrid")
	
	# Override seaborn's default sans-serif with Times New Roman
	plt.rcParams.update({
		'font.family': 'serif',
		'mathtext.fontset': 'stix',  # Makes math text match Times New Roman
		'font.size': 30,             
		'axes.labelsize': 36,        # Large, clear axis labels
		'legend.fontsize': 30,       
		'xtick.labelsize': 30,       
		'ytick.labelsize': 30,       
		# 2. FIX BORDER: Force 100% visible, solid black borders on all 4 sides
		'axes.linewidth': 2.0,       
		'axes.edgecolor': 'black',
		'axes.spines.top': True,
		'axes.spines.right': True,
		'axes.spines.bottom': True,
		'axes.spines.left': True,
		
		'grid.alpha': 0.8,           # Softer grid
		'grid.linestyle': '--'
	})

def plot_all_features(input_file, results_dir):
	os.makedirs(results_dir, exist_ok=True)
	set_plot_style()
	
	print("\n" + "="*60)
	print("🎨 GENERATING HEP-STYLE FEATURE DISTRIBUTIONS")
	print("="*60)

	# 1. Load the Data
	print(f"Loading data from: {input_file}")
	with h5py.File(input_file, 'r') as f:
		X = f['features'][:]
		y = f['labels'][:]
		feature_names = f.attrs.get('feature_names', [f"Feature_{i}" for i in range(X.shape[1])])
		
	# Convert byte strings to normal strings if necessary
	feature_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in feature_names]
	print(f"Dataset loaded: {X.shape[0]:,} jets with {X.shape[1]} features.")

	# 2. Separate Classes
	print("Separating classes into QCD (Background) and Top (Signal)...")
	mask_qcd = (y == 0)
	mask_top = (y == 1)
	
	X_qcd = X[mask_qcd]
	X_top = X[mask_top]

	# 3. Generate a plot for every single feature
	print("\nGenerating and saving plots...")
	for i, feat_name in enumerate(tqdm(feature_names, desc="Plotting Features")):
		plt.figure(figsize=(12, 8))
		
		feat_data_qcd = X_qcd[:, i]
		feat_data_top = X_top[:, i]
		
		# Robust binning: Use 1st and 99th percentiles to cut off extreme outliers
		min_val = np.percentile(X[:, i], 1)
		max_val = np.percentile(X[:, i], 99)
		bins = np.linspace(min_val, max_val, 50)
		
		# Plot Background (QCD) - Crisp 1.5 line weight
		plt.hist(feat_data_qcd, bins=bins, density=True, 
				 facecolor=to_rgba('blue', 0.8), edgecolor='blue', 
				 linewidth=1.5, histtype='stepfilled', label='QCD (Background)')
		
		# Plot Signal (Top) - Crisp 1.5 line weight
		plt.hist(feat_data_top, bins=bins, density=True, 
				 facecolor=to_rgba('red', 0.8), edgecolor='red', 
				 linewidth=1.5, histtype='stepfilled', label='Top (Signal)')
		
		# Clean Text Formatting (Removed fontweight='bold')
		plt.xlabel(feat_name, labelpad=15)
		plt.ylabel('Normalized Density', labelpad=15)
		
		# Clean legend without heavy shadows
		plt.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
		
		# Clean the filename 
		safe_name = feat_name.replace("/", "_").replace(" ", "_")
		plot_path = os.path.join(results_dir, f"dist_{safe_name}.png")
		
		plt.savefig(plot_path, bbox_inches='tight', dpi=300)
		plt.close()

	print("\n" + "="*60)
	print(f"✅ ALL {len(feature_names)} PLOTS SAVED TO:\n\t{results_dir}")
	print("="*60)

if __name__ == "__main__":
	plot_all_features(INPUT_H5, RESULTS_DIR)
