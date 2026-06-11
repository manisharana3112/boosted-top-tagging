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
H5_DATA_PATH = "/home/sachit/ML/processed_data/jetclass_1M/p_net_dataset_1M.h5"
RESULTS_DIR = "/home/sachit/ML/jetclass_Results_1M/ParticleNet"

BATCH_SIZE = 1024  # Maximize this to fully utilize RTX 4060 VRAM
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5       # Stop if Val Loss doesn't improve for 5 epochs

# ==========================================
# Early Stopping Class
# ==========================================
class EarlyStopping:
	def __init__(self, patience=5, min_delta=0.0):
		"""
		Args:
			patience (int): How many epochs to wait after last time validation loss improved.
			min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = float('inf')
		self.early_stop = False

	def __call__(self, val_loss):
		if val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
		else:
			self.counter += 1
			print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
			if self.counter >= self.patience:
				self.early_stop = True

# ==========================================
# 2. ParticleNet Dataset Loader (Fast Streaming)
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
		# Get info about the current PyTorch Worker
		worker_info = torch.utils.data.get_worker_info()
		
		q_start, q_end = self.q_range
		t_start, t_end = self.t_range
		
		# Shard the dataset if using multiple workers
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

		# Every worker gets its own independent read-only handle
		with h5py.File(self.h5_path, 'r') as f:
			qcd_idx = q_start
			top_idx = t_start
			half_chunk = self.chunk_size // 2

			while qcd_idx < q_end or top_idx < t_end:
				q_step_end = min(qcd_idx + half_chunk, q_end)
				t_step_end = min(top_idx + half_chunk, t_end)

				# Independent, sequential read for this worker's chunk
				p_buf = np.concatenate([f['points'][qcd_idx:q_step_end], f['points'][top_idx:t_step_end]], axis=0)
				f_buf = np.concatenate([f['features'][qcd_idx:q_step_end], f['features'][top_idx:t_step_end]], axis=0)
				m_buf = np.concatenate([f['mask'][qcd_idx:q_step_end], f['mask'][top_idx:t_step_end]], axis=0)
				y_buf = np.concatenate([f['labels'][qcd_idx:q_step_end], f['labels'][top_idx:t_step_end]], axis=0)

				shuffle_idx = np.random.permutation(len(y_buf))

				for i in shuffle_idx:
					yield (
						torch.tensor(p_buf[i], dtype=torch.float32),
						torch.tensor(f_buf[i], dtype=torch.float32),
						torch.tensor(m_buf[i], dtype=torch.float32),
						torch.tensor(y_buf[i], dtype=torch.float32).unsqueeze(0)
					)

				qcd_idx = q_step_end
				top_idx = t_step_end

# ==========================================
# 3. ParticleNet Architecture (EdgeConv)
# ==========================================
def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1), x)
	xx = torch.sum(x**2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1)
	idx = pairwise_distance.topk(k=k, dim=-1)[1]
	return idx

def get_graph_feature(x, k, idx=None):
	batch_size, num_dims, num_points = x.size()
	if idx is None:
		idx = knn(x, k=k)
	device = x.device

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
	idx = idx + idx_base
	idx = idx.view(-1)

	x = x.transpose(2, 1).contiguous()
	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
	
	feature = torch.cat((feature, feature - x), dim=3).permute(0, 3, 1, 2).contiguous()
	return feature

class EdgeConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, k=16):
		super(EdgeConvBlock, self).__init__()
		self.k = k
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)

	def forward(self, x, coords):
		x = get_graph_feature(x, k=self.k, idx=knn(coords, k=self.k))
		x = self.conv(x)
		x = x.max(dim=-1, keepdim=False)[0]
		return x

class ParticleNet(nn.Module):
	def __init__(self, input_dims=7, num_classes=1):
		super(ParticleNet, self).__init__()
		self.input_bn = nn.BatchNorm1d(input_dims)
		self.block1 = EdgeConvBlock(input_dims, 64)
		self.block2 = EdgeConvBlock(64, 128)
		self.block3 = EdgeConvBlock(128, 256)
		self.fc = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, num_classes)
		)

	def forward(self, points, features, mask):
		x = self.input_bn(features)
		x = self.block1(x, points)
		x = self.block2(x, points)
		x = self.block3(x, points)
		x = x * mask
		x = x.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
		return self.fc(x)

# ==========================================
# Helper: Evaluate and Extract Scores
# ==========================================
def get_model_scores(model, dataloader, device, desc="Evaluating"):
	model.eval()
	all_y, all_probs = [], []
	
	pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True, unit='batch', colour='cyan')
	
	with torch.no_grad():
		for p, f, m, y in pbar:
			p = p.to(device, non_blocking=True)
			f = f.to(device, non_blocking=True)
			m = m.to(device, non_blocking=True)
			
			with torch.amp.autocast('cuda'):
				probs = torch.sigmoid(model(p, f, m)).cpu().numpy()
				
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
	
	# Note: shuffle must be False for IterableDataset
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, prefetch_factor=2)
	
	model = ParticleNet().to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.BCEWithLogitsLoss()

	scaler = torch.amp.GradScaler('cuda')
	early_stopping = EarlyStopping(patience=PATIENCE)
	history = {'train_loss': [], 'val_loss': []}

	# Checkpointing Setup
	model_save_path = os.path.join(RESULTS_DIR, "particlenet_best_model.pth")
	print(f"Model checkpoints will be saved to: {model_save_path}\n")

	for epoch in range(EPOCHS):
		model.train()
		total_loss = 0
		
		train_batches = len(train_data) // BATCH_SIZE
		train_pbar = tqdm(train_loader, total=train_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", 
						  leave=False, dynamic_ncols=True, unit='batch', colour='green')
		
		for p, f, m, y in train_pbar:
			p = p.to(device, non_blocking=True)
			f = f.to(device, non_blocking=True)
			m = m.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			
			optimizer.zero_grad()
			
			with torch.amp.autocast('cuda'):
				out = model(p, f, m)
				loss = criterion(out, y)
			
			if torch.isnan(loss):
				continue
			
			scaler.scale(loss).backward()
			
			# Gradient Clipping
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
			for p, f, m, y in val_pbar:
				p = p.to(device, non_blocking=True)
				f = f.to(device, non_blocking=True)
				m = m.to(device, non_blocking=True)
				y = y.to(device, non_blocking=True)
				
				with torch.amp.autocast('cuda'):
					out = model(p, f, m)
					v_loss = criterion(out, y).item()
					
				val_loss += v_loss
				val_pbar.set_postfix({'val_loss': f"{v_loss:.4f}"})
				
		avg_train_loss = total_loss / max(1, train_batches)
		avg_val_loss = val_loss / max(1, val_batches)
		
		history['train_loss'].append(avg_train_loss)
		history['val_loss'].append(avg_val_loss)
		
		print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
		
		# Early Stopping and Checkpointing Check
		early_stopping(avg_val_loss)
		if early_stopping.counter == 0:
			torch.save(model.state_dict(), model_save_path)
			print(f"--> Validation loss improved! Saved model to {model_save_path}")
			
		if early_stopping.early_stop:
			print(f"\n[!] Early stopping triggered at epoch {epoch+1}. Restoring best weights...")
			model.load_dict(torch.load(model_save_path))
			break
		
	# ==========================================
	# 5. Final Evaluation & Plotting
	# ==========================================
	print("\nExtracting scores for all datasets...")
	
	# Make sure we evaluate using the best model weights
	model.load_state_dict(torch.load(model_save_path))
	
	train_y, train_probs = get_model_scores(model, train_loader, device, desc="Eval Train")
	val_y, val_probs = get_model_scores(model, val_loader, device, desc="Eval Val")
	test_y, test_probs = get_model_scores(model, test_loader, device, desc="Eval Test")

	scores_file = os.path.join(RESULTS_DIR, "particlenet_scores.npz")
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
	actual_epochs_run = len(history['train_loss'])
	plt.figure(figsize=(8, 6), dpi=300)
	epochs_range = range(1, actual_epochs_run + 1)
	plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue', lw=2)
	plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='red', lw=2)
	plt.xlabel('Epoch', weight='bold')
	plt.ylabel('BCE Loss', weight='bold')
	plt.title('ParticleNet Learning Curve', weight='bold')
	plt.legend(frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 1. Confusion Matrix
	cm = confusion_matrix(test_y, test_probs > 0.5, labels=[0, 1])
	plt.figure(figsize=(8, 6), dpi=300)
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
				xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'],
				annot_kws={"size": 16}, cbar_kws={'label': 'Count'})
	plt.title('ParticleNet Confusion Matrix (Test Set)', weight='bold')
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
	plt.title('ParticleNet ROC Curves', weight='bold')
	plt.legend(loc="lower right", frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/roc_curve_comparison.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 3. Score Distribution
	plt.figure(figsize=(8, 6), dpi=300)
	sns.histplot(test_probs[test_y == 0], bins=50, color='blue', alpha=0.5, label='QCD (Background)', stat='density', element='step', fill=True)
	sns.histplot(test_probs[test_y == 1], bins=50, color='red', alpha=0.5, label='Top (Signal)', stat='density', element='step', fill=True)
	plt.xlabel('ParticleNet Score', weight='bold')
	plt.ylabel('Density', weight='bold')
	plt.title('Classifier Score Distribution (Test Set)', weight='bold')
	plt.legend(loc='upper center')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/score_distribution_test.png", dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"Results saved to {RESULTS_DIR}")
	
if __name__ == "__main__":
	run_training()
