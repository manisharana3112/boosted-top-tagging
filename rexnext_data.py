import os
import glob
import gc
import uproot
import awkward as ak
import numpy as np
import h5py

# ==========================================
# 1. Configuration
# ==========================================
INPUT_DIR = "/home/sachit/ML/data/jetclass"
OUTPUT_DIR = "/home/sachit/ML/processed_data/test_jetclass"
MASTER_H5_PATH = os.path.join(OUTPUT_DIR, "test_rexnext_dataset.h5")

# Step size lowered slightly for CNN to prevent memory overload 
# during the heavy 2D histogram looping
STEP_SIZE = 10000

# CNN Specifics
N_PIXELS = 64
IMAGE_RANGE = 1.0

# Tree name in JetClass ROOT files
TREE_NAME = "tree"

if __name__ == "__main__":
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# PRODUCTION MODE: Grab ROOT files (using same slicing as p_net_data)
	qcd_files = sorted(glob.glob(f"{INPUT_DIR}/ZJetsToNuNu_*.root"))[:10]
	signal_files = sorted(glob.glob(f"{INPUT_DIR}/TTBar_*.root"))[:10] 
	root_files = qcd_files + signal_files

	if not root_files:
		print(f"No ROOT files found in {INPUT_DIR}!")
		exit()

	print(f"Initializing Full CNN HDF5 file at: {MASTER_H5_PATH}")
	
	# Pre-calculate bins for 2D Histogram
	bins = np.linspace(-IMAGE_RANGE, IMAGE_RANGE, N_PIXELS + 1)

	with h5py.File(MASTER_H5_PATH, 'w') as h5_file:
		
		# Initialize expandable datasets with LZF compression
		h5_images = h5_file.create_dataset('images', shape=(0, N_PIXELS, N_PIXELS), maxshape=(None, N_PIXELS, N_PIXELS), dtype=np.float32, chunks=True, compression="lzf")
		h5_labels = h5_file.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True, compression="lzf")
		
		dt_str = h5py.string_dtype(encoding='utf-8')
		h5_ids = h5_file.create_dataset('ids', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True, compression="lzf")

		grand_total_jets = 0

		branches_to_read = ["part_px", "part_py", "part_deta", "part_dphi"]

		for FILENAME in root_files:
			base_name = os.path.basename(FILENAME).replace(".root", "")
			print(f"\n{'='*60}")
			print(f"PROCESSING: {base_name}")
			print(f"{'='*60}")
			
			# Labeling logic based on filename prefix
			is_signal_file = "ZJetsToNuNu" not in base_name

			if os.path.getsize(FILENAME) == 0:
				continue

			try:
				file = uproot.open(FILENAME)
				f_r = file[TREE_NAME]
			except Exception as e:
				print(f"Error opening {base_name}: {e}")
				continue
			
			chunk_num = 0
			file_jets_saved = 0
			
			# Stream data in chunks to keep memory usage low
			for chunk in f_r.iterate(branches_to_read, library="ak", step_size=STEP_SIZE):
				
				chunk_size = len(chunk["part_px"])
				if chunk_size == 0:
					continue

				# --- CALCULATE PARTICLE KINEMATICS ---
				px = chunk["part_px"]
				py = chunk["part_py"]
				part_pt = np.sqrt(px**2 + py**2)
				part_deta = chunk["part_deta"]
				part_dphi = chunk["part_dphi"]

				# =========================================================
				# IMAGE DATA PREPARATION (Exact Landscape Logic)
				# =========================================================
				chunk_images = np.zeros((chunk_size, N_PIXELS, N_PIXELS), dtype=np.float32)
				valid_jet_mask = np.ones(chunk_size, dtype=bool)

				for i in range(chunk_size):
					pt = ak.to_numpy(part_pt[i])
					eta = ak.to_numpy(part_deta[i])
					phi = ak.to_numpy(part_dphi[i])

					if len(pt) < 2:
						valid_jet_mask[i] = False
						continue
						
					pt_sum = np.sum(pt)
					if pt_sum == 0: 
						valid_jet_mask[i] = False
						continue
						
					# 1. Center
					eta_center = np.sum(pt * eta) / pt_sum
					sum_sin = np.sum(pt * np.sin(phi))
					sum_cos = np.sum(pt * np.cos(phi))
					phi_center = np.arctan2(sum_sin, sum_cos)

					eta = eta - eta_center
					phi = phi - phi_center
					phi = (phi + np.pi) % (2 * np.pi) - np.pi

					# 2. Principal Axis Rotation
					cov = np.cov(eta, phi, aweights=pt)
					eigenvalues, eigenvectors = np.linalg.eigh(cov)
					
					principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
					theta = np.arctan2(principal_axis[1], principal_axis[0])
					rotation_angle = -theta + (np.pi / 2.0)
					
					c, s = np.cos(rotation_angle), np.sin(rotation_angle)
					eta_rot = c * eta - s * phi
					phi_rot = s * eta + c * phi

					# 3. Flipping
					top_right = np.sum(pt[(eta_rot > 0) & (phi_rot > 0)])
					top_left = np.sum(pt[(eta_rot < 0) & (phi_rot > 0)])
					bottom_right = np.sum(pt[(eta_rot > 0) & (phi_rot < 0)])
					bottom_left = np.sum(pt[(eta_rot < 0) & (phi_rot < 0)])

					if (top_left + bottom_left) > (top_right + bottom_right):
						eta_rot = -eta_rot
						
					if (bottom_left + bottom_right) > (top_left + top_right):
						phi_rot = -phi_rot

					# 4. Histogram and Normalize
					img, _, _ = np.histogram2d(eta_rot, phi_rot, bins=(bins, bins), weights=pt)
					
					norm = np.sum(img)
					if norm > 0:
						chunk_images[i] = img / norm
					else:
						valid_jet_mask[i] = False

				# Filter out invalid jets
				valid_images = chunk_images[valid_jet_mask]
				n_valid_jets = len(valid_images)
				
				if n_valid_jets == 0:
					chunk_num += 1
					continue

				# --- LABELS ---
				target_label = 1 if is_signal_file else 0
				labels = np.full(n_valid_jets, target_label, dtype=np.int64)

				# --- SAVE TO MASTER H5 ---
				global_nums = (chunk_num * STEP_SIZE) + np.arange(chunk_size)[valid_jet_mask]
				chunk_ids = np.array([f"{base_name}_jet_{idx}" for idx in global_nums], dtype=object)
				
				current_size = h5_images.shape[0]
				h5_images.resize(current_size + n_valid_jets, axis=0)
				h5_labels.resize(current_size + n_valid_jets, axis=0)
				h5_ids.resize(current_size + n_valid_jets, axis=0)
				
				h5_images[current_size:] = valid_images
				h5_labels[current_size:] = labels
				h5_ids[current_size:] = chunk_ids
				
				file_jets_saved += n_valid_jets
				chunk_num += 1
				
				print(f"   Stored {file_jets_saved:,} jets...", end='\r')
				
				# Memory Cleanup
				del chunk, px, py, part_pt, part_deta, part_dphi
				del chunk_images, valid_images, labels, chunk_ids, valid_jet_mask
				gc.collect()

			print(f"\nFinished {base_name}. Extracted {file_jets_saved:,} valid jets.")
			grand_total_jets += file_jets_saved
			file.close()

	print(f"\n{'='*60}\nALL FILES PROCESSED SUCCESSFULLY\nTotal jets: {grand_total_jets:,}\n{'='*60}")
