import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==========================================
# 1. Configuration
# ==========================================
H5_DATA_PATH = "/home/sachit/ML/processed_data/jetclass_1M/cnn_dataset_1M.h5"
OUTPUT_DIR = "/home/sachit/ML/jetclass_Results_1M/sample_jet_image"

def plot_sample_jets():
    print(f"Loading data from {H5_DATA_PATH}...")
    
    with h5py.File(H5_DATA_PATH, 'r') as f:
        labels = f['labels'][:]
        
        # Find the first index of a QCD jet (label 0) and Top jet (label 1)
        qcd_idx = np.where(labels == 0)[0][0]
        top_idx = np.where(labels == 1)[0][0]
        
        # Extract the 40x40 images
        qcd_img = f['images'][qcd_idx]
        top_img = f['images'][top_idx]

    # ==========================================
    # 2. Plotting
    # ==========================================
    # Set global font sizes for presentations
    plt.rcParams.update({
        'font.size': 14, 
        'axes.titlesize': 20, 
        'axes.labelsize': 18,
        'xtick.labelsize': 14, 
        'ytick.labelsize': 14
    })
    
    # Standard JetClass image bounds in eta-phi space
    extent = [-0.8, 0.8, -0.8, 0.8]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Use 'magma' colormap for a beautiful "energy" glow. 
    # LogNorm helps visualize faint background particles next to the bright core.
    cmap = 'magma'
    
    # Plot 1: QCD Jet
    im0 = axes[0].imshow(qcd_img, extent=extent, origin='lower', cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1.0))
    axes[0].set_title('Background: QCD Jet', weight='bold', pad=15)
    axes[0].set_xlabel(r'$\Delta \eta$', weight='bold')
    axes[0].set_ylabel(r'$\Delta \phi$', weight='bold')
    axes[0].grid(False)
    
    # Plot 2: Top Jet
    im1 = axes[1].imshow(top_img, extent=extent, origin='lower', cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1.0))
    axes[1].set_title('Signal: Top Quark Jet', weight='bold', pad=15)
    axes[1].set_xlabel(r'$\Delta \eta$', weight='bold')
    axes[1].set_ylabel(r'$\Delta \phi$', weight='bold')
    axes[1].grid(False)
    
    # Add a shared colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label('Transverse Momentum ($p_T$) Fraction', weight='bold', size=16, labelpad=15)
    
    plt.suptitle("Jet Substructure (40x40 Image Representation)", weight='bold', size=24, y=1.05)
    
    # Save the output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = os.path.join(OUTPUT_DIR, "jet_images.png")
    out_pdf = os.path.join(OUTPUT_DIR, "jet_images.pdf")
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"✅ Success! sample images saved to:")
    print(f"   - {out_png}")

if __name__ == "__main__":
    plot_sample_jets()
