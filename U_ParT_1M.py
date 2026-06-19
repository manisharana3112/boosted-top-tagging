import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import math

# ==========================================
# 1. Configuration
# ==========================================
# Update this path to point to your actual 2M (or 1M) dataset
H5_DATA_PATH = "/mnt/c/Users/ranam/Downloads/JetClass_Pythia_train_100M_part0/upart_dataset_1M/upart_dataset_1M.h5" 
RESULTS_DIR = "/mnt/c/Users/ranam/Downloads/JetClass_Pythia_train_100M_part0/upart_dataset_1M/UParT_1"

BATCH_SIZE = 256  # Drop to 512 if you hit an OOM error
EPOCHS = 20
LEARNING_RATE = 0.001

# UParT Hyperparameters
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
N_FEATURES = 17
N_INTERACTIONS = 4

# ==========================================
# 2. UParT Dataset Loader (Fast Streaming)
# ==========================================
class FastStreamingDataset(torch.utils.data.IterableDataset):
	def __init__(self, h5_path, split='train', chunk_size=4096):
		self.h5_path = h5_path
		self.chunk_size = chunk_size

		# Load just the labels to find class boundaries
		with h5py.File(h5_path, 'r') as f:
			all_labels = f['labels'][:]

		qcd_indices = np.where(all_labels == 0)[0]
		top_indices = np.where(all_labels == 1)[0]

		q_start, q_end = int(np.min(qcd_indices)), int(np.max(qcd_indices)) + 1
		t_start, t_end = int(np.min(top_indices)), int(np.max(top_indices)) + 1

		q_count = q_end - q_start
		t_count = t_end - t_start

		# Split boundaries (70% Train, 15% Val, 15% Test)
		if split == 'train':
			self.q_range = (q_start, q_start + int(q_count * 0.7))
			self.t_range = (t_start, t_start + int(t_count * 0.7))
		elif split == 'val':
			self.q_range = (q_start + int(q_count * 0.7), q_start + int(q_count * 0.85))
			self.t_range = (t_start + int(t_count * 0.7), t_start + int(t_count * 0.85))
		else: # test
			self.q_range = (q_start + int(q_count * 0.85), q_end)
			self.t_range = (t_start + int(t_count * 0.85), t_end)

		self.length = (self.q_range[1] - self.q_range[0]) + (self.t_range[1] - self.t_range[0])

	def __len__(self):
		return self.length

	def __iter__(self):
		# --- NEW: Get info about the current PyTorch Worker ---
		worker_info = torch.utils.data.get_worker_info()
		
		q_start, q_end = self.q_range
		t_start, t_end = self.t_range
		
		# --- NEW: Shard the dataset if using multiple workers ---
		if worker_info is not None:
			worker_id = worker_info.id
			num_workers = worker_info.num_workers
			
			# Slice the QCD indices for this specific worker
			q_chunk = math.ceil((q_end - q_start) / num_workers)
			q_start = q_start + (worker_id * q_chunk)
			q_end = min(q_start + q_chunk, q_end)
			
			# Slice the Top indices for this specific worker
			t_chunk = math.ceil((t_end - t_start) / num_workers)
			t_start = t_start + (worker_id * t_chunk)
			t_end = min(t_start + t_chunk, t_end)

		# By opening the file HERE, every worker gets its own independent read-only handle
		with h5py.File(self.h5_path, 'r') as f:
			qcd_idx = q_start
			top_idx = t_start
			half_chunk = self.chunk_size // 2

			while qcd_idx < q_end or top_idx < t_end:
				q_step_end = min(qcd_idx + half_chunk, q_end)
				t_step_end = min(top_idx + half_chunk, t_end)

				# Independent, sequential read for this worker's chunk
				f_buf = np.concatenate([f['features'][qcd_idx:q_step_end], f['features'][top_idx:t_step_end]], axis=0)
				i_buf = np.concatenate([f['interactions'][qcd_idx:q_step_end], f['interactions'][top_idx:t_step_end]], axis=0)
				m_buf = np.concatenate([f['mask'][qcd_idx:q_step_end], f['mask'][top_idx:t_step_end]], axis=0)
				y_buf = np.concatenate([f['labels'][qcd_idx:q_step_end], f['labels'][top_idx:t_step_end]], axis=0)

				shuffle_idx = np.random.permutation(len(y_buf))

				for i in shuffle_idx:
					yield (
						torch.tensor(f_buf[i], dtype=torch.float32),
						torch.tensor(i_buf[i], dtype=torch.float32),
						torch.tensor(m_buf[i], dtype=torch.float32),
						torch.tensor(y_buf[i], dtype=torch.float32).unsqueeze(0)
					)

				qcd_idx = q_step_end
				top_idx = t_step_end
				
# ==========================================
# 3. UParT Architecture
# ==========================================
class ParticleAttentionBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, interaction_dim=4, dropout=0.1):
		super(ParticleAttentionBlock, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		
		self.pair_proj = nn.Sequential(
			nn.Linear(interaction_dim, 64),
			nn.GELU(),
			nn.Linear(64, num_heads)
		)

		self.out_proj = nn.Linear(embed_dim, embed_dim)
		
		self.ffn = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(embed_dim * 4, embed_dim)
		)

		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)

	def forward(self, x, interactions, mask):
		B, N, D = x.size()
		
		residual = x
		x = self.norm1(x)

		q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

		pair_bias = self.pair_proj(interactions.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		scores = scores + pair_bias

		attn_mask = mask.unsqueeze(1)
		# FP16 Fix: -10000.0 instead of float('-inf') to prevent NaN crashes
		scores = scores.masked_fill(attn_mask == 0, -10000.0)

		attn = torch.softmax(scores, dim=-1)
		
		out = torch.matmul(attn, v)
		out = out.transpose(1, 2).contiguous().view(B, N, D)
		out = self.out_proj(out)
		
		x = residual + self.drop1(out)
		
		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = residual + self.drop2(x)
		
		return x

class UParT(nn.Module):
	def __init__(self, num_classes=1):
		super(UParT, self).__init__()
		
		self.input_embed = nn.Sequential(
			nn.Linear(N_FEATURES, EMBED_DIM),
			nn.GELU(),
			nn.Linear(EMBED_DIM, EMBED_DIM)
		)
		
		self.blocks = nn.ModuleList([
			ParticleAttentionBlock(EMBED_DIM, NUM_HEADS, N_INTERACTIONS)
			for _ in range(NUM_LAYERS)
		])
		
		self.norm = nn.LayerNorm(EMBED_DIM)
		
		self.fc = nn.Sequential(
			nn.Linear(EMBED_DIM, 256),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(256, num_classes)
		)

	def forward(self, features, interactions, mask):
		x = features.transpose(1, 2)
		x = self.input_embed(x)
		
		for block in self.blocks:
			x = block(x, interactions, mask)
			
		x = self.norm(x)
		
		mask_t = mask.transpose(1, 2)
		x = x * mask_t
		x = x.sum(dim=1) / (mask_t.sum(dim=1) + 1e-8)
		
		return self.fc(x)

# ==========================================
# Helper: Evaluate and Extract Scores
# ==========================================
def get_model_scores(model, dataloader, device, desc="Evaluating"):
	model.eval()
	all_y, all_probs = [], []
	
	pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True, unit='batch', colour='cyan')
	
	with torch.no_grad():
		for f, inter, m, y in pbar:
			f = f.to(device, non_blocking=True)
			inter = inter.to(device, non_blocking=True)
			m = m.to(device, non_blocking=True)
			
			with torch.amp.autocast('cuda'):
				probs = torch.sigmoid(model(f, inter, m)).cpu().numpy()
				
			all_probs.extend(probs.flatten())
			all_y.extend(y.numpy().flatten())
			
	return np.array(all_y), np.array(all_probs)

# ==========================================
# 4. Training Engine
# ==========================================
def run_training():
	os.makedirs(RESULTS_DIR, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Hardware Optimizations
	torch.backends.cudnn.benchmark = True
	print(f"Device: {device} | Using Mixed Precision (AMP) & cuDNN Benchmark")

	# Initialize Iterable Datasets
	train_data = FastStreamingDataset(H5_DATA_PATH, split='train')
	val_data = FastStreamingDataset(H5_DATA_PATH, split='val')
	test_data = FastStreamingDataset(H5_DATA_PATH, split='test')
	
	# Set num_workers=0 to prevent h5py deadlocks. Pin memory ensures fast CPU->GPU transfer.
	# Using 6 workers safely with dataset sharding
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	
	model = UParT().to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.BCEWithLogitsLoss()

	scaler = torch.amp.GradScaler('cuda')
	history = {'train_loss': [], 'val_loss': []}

	# Checkpointing Setup
	best_val_loss = float('inf')
	model_save_path = os.path.join(RESULTS_DIR, "upart_best_model.pth")
	print(f"Model checkpoints will be saved to: {model_save_path}\n")

	for epoch in range(EPOCHS):
		model.train()
		total_loss = 0
		
		# For IterableDatasets, tqdm cannot automatically know the total batches.
		# We calculate it explicitly so the progress bar works correctly.
		train_batches = len(train_data) // BATCH_SIZE
		train_pbar = tqdm(train_loader, total=train_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", 
						  leave=False, dynamic_ncols=True, unit='batch', colour='green')
		
		for f_feat, inter, m, y in train_pbar:
			f_feat = f_feat.to(device, non_blocking=True)
			inter = inter.to(device, non_blocking=True)
			m = m.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			
			optimizer.zero_grad()
			
			with torch.amp.autocast('cuda'):
				out = model(f_feat, inter, m)
				loss = criterion(out, y)
			
			# --- NEW: Catch NaNs before they corrupt the loss tracking ---
			if torch.isnan(loss):
				continue
			
			scaler.scale(loss).backward()
			
			# --- NEW: Gradient Clipping to prevent explosion ---
			# We must unscale the gradients before clipping them
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			
			scaler.step(optimizer)
			scaler.update()
			
			total_loss += loss.item()
			train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
					
		model.eval()
		val_loss = 0
		
		val_batches = len(val_data) // BATCH_SIZE
		val_pbar = tqdm(val_loader, total=val_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", 
						leave=False, dynamic_ncols=True, unit='batch', colour='blue')
		
		with torch.no_grad():
			for f_feat, inter, m, y in val_pbar:
				f_feat = f_feat.to(device, non_blocking=True)
				inter = inter.to(device, non_blocking=True)
				m = m.to(device, non_blocking=True)
				y = y.to(device, non_blocking=True)
				
				with torch.amp.autocast('cuda'):
					out = model(f_feat, inter, m)
					v_loss = criterion(out, y).item()
					
				val_loss += v_loss
				val_pbar.set_postfix({'val_loss': f"{v_loss:.4f}"})
				
		avg_train_loss = total_loss / max(1, train_batches)
		avg_val_loss = val_loss / max(1, val_batches)
		
		history['train_loss'].append(avg_train_loss)
		history['val_loss'].append(avg_val_loss)
		
		print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
		
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(model.state_dict(), model_save_path)
			print(f"--> Validation loss improved! Saved model to {model_save_path}")
		
	# ==========================================
	# 5. Final Evaluation & Plotting
	# ==========================================
	print("\nExtracting scores for all datasets...")
	train_y, train_probs = get_model_scores(model, train_loader, device, desc="Eval Train")
	val_y, val_probs = get_model_scores(model, val_loader, device, desc="Eval Val")
	test_y, test_probs = get_model_scores(model, test_loader, device, desc="Eval Test")

	scores_file = os.path.join(RESULTS_DIR, "upart_scores.npz")
	np.savez_compressed(
		scores_file,
		train_y=train_y, train_probs=train_probs,
		val_y=val_y, val_probs=val_probs,
		test_y=test_y, test_probs=test_probs
	)
	print(f"Raw scores saved successfully to: {scores_file}")

	print("\nGenerating and saving plots...")
	plt.rcParams.update({
		'font.size': 14,
		'axes.titlesize': 16,
		'axes.labelsize': 14,
		'xtick.labelsize': 12,
		'ytick.labelsize': 12,
		'legend.fontsize': 12,
		'figure.titlesize': 18
	})

	# 0. Learning Curve
	plt.figure(figsize=(8, 6), dpi=300)
	epochs_range = range(1, EPOCHS + 1)
	plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue', lw=2)
	plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='red', lw=2)
	plt.xlabel('Epoch', weight='bold')
	plt.ylabel('BCE Loss', weight='bold')
	plt.title('UParT Learning Curve', weight='bold')
	plt.legend(frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 1. Confusion Matrix
	cm = confusion_matrix(test_y, test_probs > 0.5)
	plt.figure(figsize=(8, 6), dpi=300)
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
				xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'],
				annot_kws={"size": 16}, cbar_kws={'label': 'Count'})
	plt.title('UParT Confusion Matrix (Test Set)', weight='bold')
	plt.xlabel('Predicted Label', weight='bold')
	plt.ylabel('True Label', weight='bold')
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/confusion_matrix_test.png", dpi=300, bbox_inches='tight')
	plt.close()
	
	# 2. ROC Curve
	plt.figure(figsize=(8, 6), dpi=300)
	fpr_tr, tpr_tr, _ = roc_curve(train_y, train_probs)
	plt.plot(fpr_tr, tpr_tr, color='blue', lw=2, linestyle=':', label=f'Train (AUC = {auc(fpr_tr, tpr_tr):.4f})')
	fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
	plt.plot(fpr_v, tpr_v, color='green', lw=2, linestyle='-.', label=f'Val (AUC = {auc(fpr_v, tpr_v):.4f})')
	fpr_t, tpr_t, _ = roc_curve(test_y, test_probs)
	plt.plot(fpr_t, tpr_t, color='darkorange', lw=2.5, label=f'Test (AUC = {auc(fpr_t, tpr_t):.4f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate (QCD Mistag)', weight='bold')
	plt.ylabel('True Positive Rate (Top Efficiency)', weight='bold')
	plt.title('UParT ROC Curves', weight='bold')
	plt.legend(loc="lower right", frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/roc_curve_comparison.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 3. Score Distribution
	plt.figure(figsize=(8, 6), dpi=300)
	sns.histplot(test_probs[test_y == 0], bins=50, color='blue', alpha=0.5, label='QCD (Background)', stat='density', element='step', fill=True)
	sns.histplot(test_probs[test_y == 1], bins=50, color='red', alpha=0.5, label='Top (Signal)', stat='density', element='step', fill=True)
	plt.xlabel('UParT Score', weight='bold')
	plt.ylabel('Density', weight='bold')
	plt.title('Classifier Score Distribution (Test Set)', weight='bold')
	plt.legend(loc='upper center')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/score_distribution_test.png", dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"Publication-ready results saved to {RESULTS_DIR}")
	
if __name__ == "__main__":
	run_training()
