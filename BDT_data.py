import os
import glob
import gc
import uproot
import awkward as ak
import numpy as np
import fastjet
import vector
import h5py

# Register vector behavior for 4-vector calculations
vector.register_awkward()

# ==========================================
# 1. Configuration
# ==========================================
INPUT_DIR = "/home/sachit/ML/data/jetclass"
OUTPUT_DIR = "/home/sachit/ML/processed_data/jetclass_1M"
# Updated filename to reflect optimized feature set
MASTER_H5_PATH = os.path.join(OUTPUT_DIR, "BDT_optimized_dataset.h5")

SUBJET_RADIUS = 0.2	# Small radius to resolve the prongs
MIN_SUBJET_PT = 5.0	# Filter out soft radiation
MIN_PT_CUT = 400
STEP_SIZE = 50000

# Optimized feature list (19 features total)
FEATURE_NAMES = [
	"n_clusters", "max_cluster_pt", "mean_cluster_pt", "std_cluster_pt",
	"max_cluster_size", "mean_cluster_size", "std_cluster_size", "total_pt",
	"cluster_pt_ratio", "cluster_size_ratio",
	# Local Coordinate Features (Good discriminators)
	"max_cluster_dr", "mean_cluster_dr", "std_cluster_dr",
	"max_cluster_deta", "mean_cluster_deta", "std_cluster_deta",
	"max_cluster_dphi", "mean_cluster_dphi", "std_cluster_dphi"
]

def extract_subjet_stats(constituents):
	"""
	Clusters constituents into sub-jets and calculates optimized 
	statistical features relative to the jet axis.
	"""
	# 1. Calculate the Jet Axis (Sum of all constituents)
	jet_v = ak.sum(constituents, axis=1)
	jet_eta = jet_v.eta
	jet_phi = jet_v.phi

	# 2. Sub-cluster the jet
	subjet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, SUBJET_RADIUS)
	cluster = fastjet.ClusterSequence(constituents, subjet_def)
	subjets = cluster.inclusive_jets(min_pt=MIN_SUBJET_PT)
	
	s_pt = subjets.pt
	s_eta = subjets.eta
	s_phi = subjets.phi
	s_mass = subjets.mass
	
	n_clusters = ak.num(s_pt)
	total_pt = ak.sum(s_pt, axis=1)
	total_size = ak.sum(s_mass, axis=1)

	# Handle empty jets to avoid NaN errors
	safe_mask = n_clusters > 0
	
	# --- Global Kinematics ---
	max_cluster_pt = ak.where(safe_mask, ak.max(s_pt, axis=1), 0.0)
	mean_cluster_pt = ak.where(safe_mask, ak.mean(s_pt, axis=1), 0.0)
	std_cluster_pt = ak.where(safe_mask, ak.std(s_pt, axis=1), 0.0)

	max_cluster_size = ak.where(safe_mask, ak.max(s_mass, axis=1), 0.0)
	mean_cluster_size = ak.where(safe_mask, ak.mean(s_mass, axis=1), 0.0)
	std_cluster_size = ak.where(safe_mask, ak.std(s_mass, axis=1), 0.0)

	# --- Ratios ---
	safe_total_pt = ak.where(total_pt == 0, 1.0, total_pt)
	safe_total_size = ak.where(total_size == 0, 1.0, total_size)
	cluster_pt_ratio = max_cluster_pt / safe_total_pt
	cluster_size_ratio = max_cluster_size / safe_total_size

	# --- Local Coordinate Calculations (Relative to Jet Axis) ---
	# deta = cluster_eta - jet_eta
	deta = s_eta - jet_eta
	
	# dphi = cluster_phi - jet_phi (with 2*pi wrap-around)
	dphi = s_phi - jet_phi
	dphi = ak.where(dphi > np.pi, dphi - 2*np.pi, dphi)
	dphi = ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
	
	# dr = sqrt(deta^2 + dphi^2)
	dr = np.sqrt(deta**2 + dphi**2)

	# Local Stats
	max_cluster_dr = ak.where(safe_mask, ak.max(dr, axis=1), 0.0)
	mean_cluster_dr = ak.where(safe_mask, ak.mean(dr, axis=1), 0.0)
	std_cluster_dr = ak.where(safe_mask, ak.std(dr, axis=1), 0.0)

	abs_deta = np.abs(deta)
	abs_dphi = np.abs(dphi)

	max_cluster_deta = ak.where(safe_mask, ak.max(abs_deta, axis=1), 0.0)
	mean_cluster_deta = ak.where(safe_mask, ak.mean(abs_deta, axis=1), 0.0)
	std_cluster_deta = ak.where(safe_mask, ak.std(deta, axis=1), 0.0)

	max_cluster_dphi = ak.where(safe_mask, ak.max(abs_dphi, axis=1), 0.0)
	mean_cluster_dphi = ak.where(safe_mask, ak.mean(abs_dphi, axis=1), 0.0)
	std_cluster_dphi = ak.where(safe_mask, ak.std(dphi, axis=1), 0.0)

	return np.column_stack([
		ak.to_numpy(n_clusters), ak.to_numpy(max_cluster_pt), ak.to_numpy(mean_cluster_pt), ak.to_numpy(std_cluster_pt),
		ak.to_numpy(max_cluster_size), ak.to_numpy(mean_cluster_size), ak.to_numpy(std_cluster_size), ak.to_numpy(total_pt),
		ak.to_numpy(cluster_pt_ratio), ak.to_numpy(cluster_size_ratio),
		# Local features only
		ak.to_numpy(max_cluster_dr), ak.to_numpy(mean_cluster_dr), ak.to_numpy(std_cluster_dr),
		ak.to_numpy(max_cluster_deta), ak.to_numpy(mean_cluster_deta), ak.to_numpy(std_cluster_deta),
		ak.to_numpy(max_cluster_dphi), ak.to_numpy(mean_cluster_dphi), ak.to_numpy(std_cluster_dphi)
	]).astype(np.float32)

if __name__ == "__main__":
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	
	top_files = sorted(glob.glob(os.path.join(INPUT_DIR, "TTBar_*.root")))[:10]
	qcd_files = sorted(glob.glob(os.path.join(INPUT_DIR, "ZJetsToNuNu_*.root")))[:10]
	root_files = top_files + qcd_files

	if not root_files:
		print(f"No ROOT files found in {INPUT_DIR}!")
		exit()

	print(f"Initializing Optimized BDT HDF5 file at: {MASTER_H5_PATH}")
	with h5py.File(MASTER_H5_PATH, 'w') as h5_file:
		# Updated shape to 19 features
		h5_features = h5_file.create_dataset('features', shape=(0, 19), maxshape=(None, 19), dtype='f4', chunks=True, compression="lzf")
		h5_labels = h5_file.create_dataset('labels', shape=(0,), maxshape=(None,), dtype='i8', chunks=True, compression="lzf")
		
		dt_str = h5py.string_dtype(encoding='utf-8')
		h5_ids = h5_file.create_dataset('ids', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True, compression="lzf")
		h5_file.attrs['feature_names'] = FEATURE_NAMES

		for FILENAME in root_files:
			base_name = os.path.basename(FILENAME).replace(".root", "")
			print(f"\n--- PROCESSING: {base_name} ---")
			is_signal_file = "TTBar_" in base_name

			try:
				with uproot.open(FILENAME) as file:
					tree = file["tree"]
					branches = ["part_px", "part_py", "part_pz", "part_energy"]
					
					chunk_num = 0
					for chunk in tree.iterate(branches, library="ak", step_size=STEP_SIZE):
						constituents = ak.zip({
							"px": chunk["part_px"], "py": chunk["part_py"],
							"pz": chunk["part_pz"], "energy": chunk["part_energy"]
						}, with_name="Momentum4D")
						
						features = extract_subjet_stats(constituents)
						n_jets = len(features)
						labels = np.full(n_jets, 1 if is_signal_file else 0, dtype='i8')
						
						global_nums = (chunk_num * STEP_SIZE) + np.arange(n_jets)
						ids = np.array([f"{base_name}_jet_{idx}" for idx in global_nums], dtype=object)
						
						curr = h5_features.shape[0]
						h5_features.resize(curr + n_jets, axis=0)
						h5_labels.resize(curr + n_jets, axis=0)
						h5_ids.resize(curr + n_jets, axis=0)
						
						h5_features[curr:] = features
						h5_labels[curr:] = labels
						h5_ids[curr:] = ids
						
						chunk_num += 1
						print(f"   Stored {h5_features.shape[0]:,} total jets...", end='\r')
						gc.collect()
			except Exception as e:
				print(f"Error processing {base_name}: {e}")

	print(f"\n\n✅ SUCCESS: Optimized BDT dataset generated with 19 features.")
	print(f"Location: {MASTER_H5_PATH}")
