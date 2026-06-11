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
OUTPUT_DIR = "/home/sachit/ML/processed_data/jetclass_1M"
MASTER_H5_PATH = os.path.join(OUTPUT_DIR, "p_net_dataset_1M.h5")

STEP_SIZE = 50000

# ParticleNet Specifics
MAX_PARTICLES = 50   
N_FEATURES = 7       # log(pt_rel), log(e_rel), log(pt), log(e), deta, dphi, dr
N_POINTS = 2         # deta, dphi

# Tree name in JetClass ROOT files
TREE_NAME = "tree"

def pad_array(arr, length=MAX_PARTICLES):
	"""Helper function to pad/clip awkward arrays to a fixed length and fill with 0.0"""
	return ak.fill_none(ak.pad_none(arr, length, clip=True), 0.0)

if __name__ == "__main__":
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# PRODUCTION MODE: Grab ROOT files
	qcd_files = sorted(glob.glob(f"{INPUT_DIR}/ZJetsToNuNu_*.root"))[:10]
	# Assuming signal files are TTBar based on your log. Adjust if necessary.
	signal_files = sorted(glob.glob(f"{INPUT_DIR}/TTBar_*.root"))[:10] 
	root_files = qcd_files + signal_files

	if not root_files:
		print(f"No ROOT files found in {INPUT_DIR}!")
		exit()

	print(f"Initializing Full ParticleNet HDF5 file at: {MASTER_H5_PATH}")
	with h5py.File(MASTER_H5_PATH, 'w') as h5_file:
		
		# Initialize expandable datasets with LZF compression
		h5_points = h5_file.create_dataset('points', shape=(0, N_POINTS, MAX_PARTICLES), maxshape=(None, N_POINTS, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		h5_features = h5_file.create_dataset('features', shape=(0, N_FEATURES, MAX_PARTICLES), maxshape=(None, N_FEATURES, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		h5_mask = h5_file.create_dataset('mask', shape=(0, 1, MAX_PARTICLES), maxshape=(None, 1, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		
		h5_labels = h5_file.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True, compression="lzf")
		dt_str = h5py.string_dtype(encoding='utf-8')
		h5_ids = h5_file.create_dataset('ids', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True, compression="lzf")

		grand_total_jets = 0

		branches_to_read = [
			"part_px", "part_py", "part_pz", "part_energy", 
			"part_deta", "part_dphi",
			"jet_pt", "jet_energy"
		]

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
				
				chunk_size = len(chunk["jet_pt"])
				if chunk_size == 0:
					continue

				# --- CALCULATE PARTICLE KINEMATICS ---
				# JetClass gives px, py. Calculate pt.
				part_pt = np.sqrt(chunk["part_px"]**2 + chunk["part_py"]**2)
				
				# Sort particles inside each jet by pT (descending)
				pt_sort_indices = ak.argsort(part_pt, axis=1, ascending=False)
				
				# Apply sorting to all constituent branches
				sorted_pt = part_pt[pt_sort_indices]
				sorted_energy = chunk["part_energy"][pt_sort_indices]
				sorted_deta = chunk["part_deta"][pt_sort_indices]
				sorted_dphi = chunk["part_dphi"][pt_sort_indices]

				# =========================================================
				# PARTICLENET DATA PREPARATION
				# =========================================================
				
				# Calculate Features
				dr = np.sqrt(sorted_deta**2 + sorted_dphi**2)
				
				log_pt = np.log(ak.where(sorted_pt > 0, sorted_pt, 1e-8))
				log_e = np.log(ak.where(sorted_energy > 0, sorted_energy, 1e-8))
				
				# Relational features (broadcast jet_pt and jet_energy to particles)
				pt_rel = sorted_pt / chunk["jet_pt"]
				e_rel = sorted_energy / chunk["jet_energy"]
				
				log_pt_rel = np.log(ak.where(pt_rel > 0, pt_rel, 1e-8))
				log_e_rel = np.log(ak.where(e_rel > 0, e_rel, 1e-8))
				
				# Create Point Cloud Mask
				mask = ak.ones_like(sorted_pt)
				
				# Pad and Clip to Fixed Length
				deta_pad = pad_array(sorted_deta)
				dphi_pad = pad_array(sorted_dphi)
				dr_pad = pad_array(dr)
				log_pt_pad = pad_array(log_pt)
				log_e_pad = pad_array(log_e)
				log_pt_rel_pad = pad_array(log_pt_rel)
				log_e_rel_pad = pad_array(log_e_rel)
				mask_pad = pad_array(mask)
				
				# Stack into Final Arrays
				chunk_points = np.stack([
					ak.to_numpy(deta_pad), 
					ak.to_numpy(dphi_pad)
				], axis=1).astype(np.float32)
				
				chunk_features = np.stack([
					ak.to_numpy(log_pt_rel_pad), 
					ak.to_numpy(log_e_rel_pad),
					ak.to_numpy(log_pt_pad), 
					ak.to_numpy(log_e_pad),
					ak.to_numpy(deta_pad), 
					ak.to_numpy(dphi_pad), 
					ak.to_numpy(dr_pad)
				], axis=1).astype(np.float32)
				
				chunk_mask = np.stack([ak.to_numpy(mask_pad)], axis=1).astype(np.float32)

				# --- LABELS ---
				if is_signal_file:
					labels = np.ones(chunk_size, dtype=np.int64)
				else:
					labels = np.zeros(chunk_size, dtype=np.int64)

				# --- SAVE TO MASTER H5 ---
				global_nums = (chunk_num * STEP_SIZE) + np.arange(chunk_size)
				chunk_ids = np.array([f"{base_name}_jet_{idx}" for idx in global_nums], dtype=object)
				
				current_size = h5_points.shape[0]
				h5_points.resize(current_size + chunk_size, axis=0)
				h5_features.resize(current_size + chunk_size, axis=0)
				h5_mask.resize(current_size + chunk_size, axis=0)
				h5_labels.resize(current_size + chunk_size, axis=0)
				h5_ids.resize(current_size + chunk_size, axis=0)
				
				h5_points[current_size:] = chunk_points
				h5_features[current_size:] = chunk_features
				h5_mask[current_size:] = chunk_mask
				h5_labels[current_size:] = labels
				h5_ids[current_size:] = chunk_ids
				
				file_jets_saved += chunk_size
				chunk_num += 1
				
				# Memory Cleanup
				del chunk, part_pt, pt_sort_indices, sorted_pt, sorted_energy, sorted_deta, sorted_dphi
				del dr, log_pt, log_e, pt_rel, e_rel, log_pt_rel, log_e_rel, mask
				del deta_pad, dphi_pad, dr_pad, log_pt_pad, log_e_pad, log_pt_rel_pad, log_e_rel_pad, mask_pad
				del chunk_points, chunk_features, chunk_mask, labels, chunk_ids
				gc.collect()

			print(f"Finished {base_name}. Extracted {file_jets_saved} valid jets.")
			grand_total_jets += file_jets_saved
			file.close()

	print(f"\n{'='*60}\nALL FILES PROCESSED SUCCESSFULLY\nTotal jets: {grand_total_jets}\n{'='*60}")
