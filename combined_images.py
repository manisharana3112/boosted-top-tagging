import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==========================================
# Configuration
# ==========================================
# Path to the dataset we just created
H5_PATH = "/home/sachit/ML/processed_data/test_jetclass/test_cnn_dataset.h5"
RESULTS_DIR = "/home/sachit/ML/test_jetclass_Results/Plots"
N_IMAGES = 10000

def plot_average_jets():
	print(f"Loading data from {H5_PATH}...")
	
	with h5py.File(H5_PATH, 'r') as f:
		labels = f['labels'][:]
		
		# Get indices for the first 10,000 signal and background jets
		sig_indices = np.where(labels == 1)[0][:N_IMAGES]
		bkg_indices = np.where(labels == 0)[0][:N_IMAGES]
		
		if len(sig_indices) == 0 or len(bkg_indices) == 0:
			print("❌ ERROR: Could not find enough images for one or both classes.")
			return

		print(f"Averaging {len(sig_indices)} Signal and {len(bkg_indices)} Background images...")
		
		# Extract the images using the indices
		# We sort indices because h5py requires monotonically increasing indices for fancy indexing
		sig_images = f['images'][np.sort(sig_indices)]
		bkg_images = f['images'][np.sort(bkg_indices)]
		
	# Calculate the mean across the N images
	avg_sig = np.mean(sig_images, axis=0)
	avg_bkg = np.mean(bkg_images, axis=0)

	# Transpose so eta is on the x-axis
	avg_sig = avg_sig.T
	avg_bkg = avg_bkg.T

	# Mask zero values so LogNorm doesn't throw a warning/error
	avg_sig = np.ma.masked_where(avg_sig <= 0, avg_sig)
	avg_bkg = np.ma.masked_where(avg_bkg <= 0, avg_bkg)

	# ==========================================
	# Formatting
	# ==========================================
	plt.rcParams.update({
		'font.size': 14, 
		'axes.titlesize': 20, 
		'axes.labelsize': 18,
		'xtick.labelsize': 14, 
		'ytick.labelsize': 14
	})

	# Set physical bounds instead of pixel indices
	physical_extent = [-0.8, 0.8, -0.8, 0.8]

	fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

	# --- Signal Plot ---
	im_sig = axs[0].imshow(avg_sig, origin='lower', extent=physical_extent, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=np.max(avg_sig)))
	axs[0].set_title('Signal: Average Top Quark', weight='bold', pad=15)
	axs[0].set_xlabel(r'$\Delta \eta$', weight='bold')
	axs[0].set_ylabel(r'$\Delta \phi$', weight='bold')
	
	# --- Background Plot ---
	im_bkg = axs[1].imshow(avg_bkg, origin='lower', extent=physical_extent, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=np.max(avg_bkg)))
	axs[1].set_title('Background: Average QCD Jet', weight='bold', pad=15)
	axs[1].set_xlabel(r'$\Delta \eta$', weight='bold')
	axs[1].set_ylabel(r'$\Delta \phi$', weight='bold')
	
	# Add a shared, formatted colorbar
	cbar = fig.colorbar(im_bkg, ax=axs.ravel().tolist(), fraction=0.02, pad=0.04)
	cbar.set_label('Average Transverse Momentum ($p_T$) Fraction', weight='bold', size=16, labelpad=15)

	plt.suptitle(f"Average Jet Substructure ({N_IMAGES:,} Jets)", weight='bold', size=24, y=1.05)

	os.makedirs(RESULTS_DIR, exist_ok=True)
	
	# Save both PNG for slides and PDF for LaTeX reports
	output_png = os.path.join(RESULTS_DIR, "average_jets.png")
	output_pdf = os.path.join(RESULTS_DIR, "average_jets.pdf")
	
	plt.savefig(output_png, dpi=300, bbox_inches='tight')
	plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
	plt.close()
	
	print(f"✅ Success! plots saved to {RESULTS_DIR}")

if __name__ == "__main__":
	plot_average_jets()
