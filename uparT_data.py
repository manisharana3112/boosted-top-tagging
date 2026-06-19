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
INPUT_DIR = "/mnt/c/Users/ranam/Downloads/JetClass_Pythia_train_100M_part0"
OUTPUT_DIR = "/mnt/c/Users/ranam/Downloads/JetClass_Pythia_train_100M_part0/upart_dataset_1M"
MASTER_H5_PATH = os.path.join(OUTPUT_DIR, "upart_dataset_1M.h5")

STEP_SIZE = 25000

# UParT Specifics
MAX_PARTICLES = 50   
N_FEATURES = 17
N_INTERACTIONS = 4

TREE_NAME = "tree"

def pad_array(arr, length=MAX_PARTICLES):
	"""Helper function to pad/clip awkward arrays to a fixed length and fill with 0.0"""
	return ak.fill_none(ak.pad_none(arr, length, clip=True), 0.0)

def calc_interaction_matrix(pt, eta, phi, e, px, py, pz):
	"""Calculates the NxN pairwise interaction tensors using broadcasting."""
	# Expand dims to create (Batch, N, 1) and (Batch, 1, N) for broadcasting
	pt_a, pt_b = np.expand_dims(pt, axis=2), np.expand_dims(pt, axis=1)
	eta_a, eta_b = np.expand_dims(eta, axis=2), np.expand_dims(eta, axis=1)
	phi_a, phi_b = np.expand_dims(phi, axis=2), np.expand_dims(phi, axis=1)
	e_a, e_b = np.expand_dims(e, axis=2), np.expand_dims(e, axis=1)
	px_a, px_b = np.expand_dims(px, axis=2), np.expand_dims(px, axis=1)
	py_a, py_b = np.expand_dims(py, axis=2), np.expand_dims(py, axis=1)
	pz_a, pz_b = np.expand_dims(pz, axis=2), np.expand_dims(pz, axis=1)

	# 1. Angular separation (Delta_ab)
	delta_ab = np.sqrt((eta_a - eta_b)**2 + (phi_a - phi_b)**2)

	# 2. Relative transverse momentum (k_T)
	min_pt = np.minimum(pt_a, pt_b)
	k_T = min_pt * delta_ab

	# 3. Momentum balance (z)
	z = min_pt / (pt_a + pt_b + 1e-8)

	# 4. Pairwise invariant mass squared (m^2)
	p_sum_sq = (px_a + px_b)**2 + (py_a + py_b)**2 + (pz_a + pz_b)**2
	m_sq = (e_a + e_b)**2 - p_sum_sq
	m_sq = np.maximum(m_sq, 0.0) # Clip negative float precision errors to 0

	# Stack into shape (Batch, 4, N, N)
	return np.stack([delta_ab, k_T, z, m_sq], axis=1).astype(np.float32)

if __name__ == "__main__":
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# Grab ROOT files
	qcd_files = sorted(glob.glob(f"{INPUT_DIR}/ZJetsToNuNu_*.root"))[:10]
	signal_files = sorted(glob.glob(f"{INPUT_DIR}/TTBar_*.root"))[:10] 
	root_files = qcd_files + signal_files

	if not root_files:
		print(f"No ROOT files found in {INPUT_DIR}!")
		exit()

	print(f"Initializing UParT HDF5 file at: {MASTER_H5_PATH}")
	with h5py.File(MASTER_H5_PATH, 'w') as h5_file:
		
		# Initialize datasets
		h5_features = h5_file.create_dataset('features', shape=(0, N_FEATURES, MAX_PARTICLES), maxshape=(None, N_FEATURES, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		h5_interactions = h5_file.create_dataset('interactions', shape=(0, N_INTERACTIONS, MAX_PARTICLES, MAX_PARTICLES), maxshape=(None, N_INTERACTIONS, MAX_PARTICLES, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		h5_mask = h5_file.create_dataset('mask', shape=(0, 1, MAX_PARTICLES), maxshape=(None, 1, MAX_PARTICLES), dtype=np.float32, chunks=True, compression="lzf")
		
		h5_labels = h5_file.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True, compression="lzf")
		dt_str = h5py.string_dtype(encoding='utf-8')
		h5_ids = h5_file.create_dataset('ids', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True, compression="lzf")

		grand_total_jets = 0

		branches_to_read = [
			"part_px", "part_py", "part_pz", "part_energy", 
			"part_deta", "part_dphi",
			"part_d0val", "part_d0err", "part_dzval", "part_dzerr",
			"part_charge", "part_isChargedHadron", "part_isNeutralHadron", 
			"part_isPhoton", "part_isElectron", "part_isMuon",
			"jet_pt", "jet_energy"
		]

		for FILENAME in root_files:
			base_name = os.path.basename(FILENAME).replace(".root", "")
			print(f"\n{'='*60}")
			print(f"PROCESSING: {base_name}")
			print(f"{'='*60}")
			
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
			
			for chunk in f_r.iterate(branches_to_read, library="ak", step_size=STEP_SIZE):
				chunk_size = len(chunk["jet_pt"])
				if chunk_size == 0: continue

				# --- 1. KINEMATICS & SORTING ---
				part_pt = np.sqrt(chunk["part_px"]**2 + chunk["part_py"]**2)
				pt_sort_indices = ak.argsort(part_pt, axis=1, ascending=False)
				
				# Apply sorting to ALL constituent branches
				s_pt = part_pt[pt_sort_indices]
				s_e = chunk["part_energy"][pt_sort_indices]
				s_px = chunk["part_px"][pt_sort_indices]
				s_py = chunk["part_py"][pt_sort_indices]
				s_pz = chunk["part_pz"][pt_sort_indices]
				s_deta = chunk["part_deta"][pt_sort_indices]
				s_dphi = chunk["part_dphi"][pt_sort_indices]

				# --- 2. CALCULATE LOG FEATURES ---
				dr = np.sqrt(s_deta**2 + s_dphi**2)
				log_pt = np.log(ak.where(s_pt > 0, s_pt, 1e-8))
				log_e = np.log(ak.where(s_e > 0, s_e, 1e-8))
				
				pt_rel = s_pt / chunk["jet_pt"]
				e_rel = s_e / chunk["jet_energy"]
				log_pt_rel = np.log(ak.where(pt_rel > 0, pt_rel, 1e-8))
				log_e_rel = np.log(ak.where(e_rel > 0, e_rel, 1e-8))
				
				mask = ak.ones_like(s_pt)

				# --- 3. PAD ALL ARRAYS TO MAX_PARTICLES ---
				pad_dict = {
					'log_pt_rel': ak.to_numpy(pad_array(log_pt_rel)),
					'log_e_rel': ak.to_numpy(pad_array(log_e_rel)),
					'log_pt': ak.to_numpy(pad_array(log_pt)),
					'log_e': ak.to_numpy(pad_array(log_e)),
					'deta': ak.to_numpy(pad_array(s_deta)),
					'dphi': ak.to_numpy(pad_array(s_dphi)),
					'dr': ak.to_numpy(pad_array(dr)),
					'd0val': ak.to_numpy(pad_array(chunk["part_d0val"][pt_sort_indices])),
					'd0err': ak.to_numpy(pad_array(chunk["part_d0err"][pt_sort_indices])),
					'dzval': ak.to_numpy(pad_array(chunk["part_dzval"][pt_sort_indices])),
					'dzerr': ak.to_numpy(pad_array(chunk["part_dzerr"][pt_sort_indices])),
					'charge': ak.to_numpy(pad_array(chunk["part_charge"][pt_sort_indices])),
					'isCH': ak.to_numpy(pad_array(chunk["part_isChargedHadron"][pt_sort_indices])),
					'isNH': ak.to_numpy(pad_array(chunk["part_isNeutralHadron"][pt_sort_indices])),
					'isPh': ak.to_numpy(pad_array(chunk["part_isPhoton"][pt_sort_indices])),
					'isEl': ak.to_numpy(pad_array(chunk["part_isElectron"][pt_sort_indices])),
					'isMu': ak.to_numpy(pad_array(chunk["part_isMuon"][pt_sort_indices])),
					
					# Variables needed for interaction matrix
					'pt': ak.to_numpy(pad_array(s_pt)),
					'e': ak.to_numpy(pad_array(s_e)),
					'px': ak.to_numpy(pad_array(s_px)),
					'py': ak.to_numpy(pad_array(s_py)),
					'pz': ak.to_numpy(pad_array(s_pz)),
					'mask': ak.to_numpy(pad_array(mask))
				}

				# --- 4. BUILD TENSORS ---
				chunk_features = np.stack([
					pad_dict['log_pt_rel'], pad_dict['log_e_rel'], 
					pad_dict['log_pt'], pad_dict['log_e'],
					pad_dict['deta'], pad_dict['dphi'], pad_dict['dr'],
					pad_dict['d0val'], pad_dict['d0err'], 
					pad_dict['dzval'], pad_dict['dzerr'],
					pad_dict['charge'], pad_dict['isCH'], pad_dict['isNH'], 
					pad_dict['isPh'], pad_dict['isEl'], pad_dict['isMu']
				], axis=1).astype(np.float32)

				chunk_interactions = calc_interaction_matrix(
					pad_dict['pt'], pad_dict['deta'], pad_dict['dphi'], 
					pad_dict['e'], pad_dict['px'], pad_dict['py'], pad_dict['pz']
				)

				chunk_mask = np.stack([pad_dict['mask']], axis=1).astype(np.float32)

				# --- LABELS ---
				labels = np.ones(chunk_size, dtype=np.int64) if is_signal_file else np.zeros(chunk_size, dtype=np.int64)

				# --- 5. SAVE TO MASTER H5 ---
				global_nums = (chunk_num * STEP_SIZE) + np.arange(chunk_size)
				chunk_ids = np.array([f"{base_name}_jet_{idx}" for idx in global_nums], dtype=object)
				
				current_size = h5_features.shape[0]
				h5_features.resize(current_size + chunk_size, axis=0)
				h5_interactions.resize(current_size + chunk_size, axis=0)
				h5_mask.resize(current_size + chunk_size, axis=0)
				h5_labels.resize(current_size + chunk_size, axis=0)
				h5_ids.resize(current_size + chunk_size, axis=0)
				
				h5_features[current_size:] = chunk_features
				h5_interactions[current_size:] = chunk_interactions
				h5_mask[current_size:] = chunk_mask
				h5_labels[current_size:] = labels
				h5_ids[current_size:] = chunk_ids
				
				file_jets_saved += chunk_size
				chunk_num += 1
				
				del chunk, pad_dict, chunk_features, chunk_interactions, chunk_mask, labels, chunk_ids
				gc.collect()

			print(f"Finished {base_name}. Extracted {file_jets_saved} valid jets.")
			grand_total_jets += file_jets_saved
			file.close()

	print(f"\n{'='*60}\nALL FILES PROCESSED SUCCESSFULLY\nTotal jets: {grand_total_jets}\n{'='*60}")
