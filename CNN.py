import os
import h5py
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
H5_DATA_PATH = "/home/sachit/ML/processed_data/jetclass_1M/cnn_dataset_1M.h5"
RESULTS_DIR = "/home/sachit/ML/jetclass_Results_1M/CNN_Opt"

BATCH_SIZE = 1024
EPOCHS = 30 # Increased slightly to give the LR scheduler room to work
LEARNING_RATE = 0.0001

# ==========================================
# 2. Float16 RAM-Safe Dataset Loader
# ==========================================
class RamSafeCNNDataset(Dataset):
	def __init__(self, h5_path):
		self.h5_path = h5_path
		
		with h5py.File(h5_path, 'r') as f:
			self.length = f['labels'].shape[0]
			print(f"🚀 Safely unpacking {self.length:,} jets into RAM (Float16)...")
			
			# Pre-allocate exact memory (keeps footprint under ~3.5 GB)
			self.images = np.empty((self.length, 40, 40), dtype=np.float16)
			self.labels = np.empty((self.length,), dtype=np.int8)
			
			chunk_size = 50000
			for i in range(0, self.length, chunk_size):
				end_idx = min(i + chunk_size, self.length)
				# Cast to float16 instantly to prevent RAM spikes
				self.images[i:end_idx] = f['images'][i:end_idx].astype(np.float16)
				self.labels[i:end_idx] = f['labels'][i:end_idx].astype(np.int8)
				
		print("✅ Dataset successfully loaded! Disk access is now bypassed.")

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		# Cast back to float32 on the fly for the GPU
		img = torch.tensor(self.images[idx].astype(np.float32)).unsqueeze(0)
		y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
		return img, y

# ==========================================
# 3. Standard CNN Architecture
# ==========================================
class JetCNN(nn.Module):
	def __init__(self, num_classes=1):
		super(JetCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2), 
			
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2), 
			
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2)  
		)
		self.classifier = nn.Sequential(
			nn.Dropout(0.3), # Slightly reduced dropout to help learning
			nn.Linear(128 * 5 * 5, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1) 
		return self.classifier(x)

# ==========================================
# Helper: Extract Scores
# ==========================================
def get_model_scores(model, dataloader, device, desc="Evaluating"):
	model.eval()
	all_y, all_probs = [], []
	pbar = tqdm(dataloader, desc=desc, leave=False)
	with torch.no_grad():
		for img, y in pbar:
			img = img.to(device, non_blocking=True)
			probs = torch.sigmoid(model(img)).cpu().numpy()
			all_probs.extend(probs.flatten())
			all_y.extend(y.numpy().flatten())
	return np.array(all_y), np.array(all_probs)

# ==========================================
# 4. Training Loop with Scheduling & Augmentation
# ==========================================
def run_training():
	os.makedirs(RESULTS_DIR, exist_ok=True)
	device = torch.device("cuda")
	torch.backends.cudnn.benchmark = True
	print(f"✅ GPU Engaged: {torch.cuda.get_device_name(0)}")

	dataset = RamSafeCNNDataset(H5_DATA_PATH)
	total_size = len(dataset)
	train_size = int(0.7 * total_size)
	val_size = int(0.15 * total_size)
	test_size = total_size - train_size - val_size
	
	train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
	
	# Workers=2 is optimal when data is fully in RAM
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
	eval_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
	val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

	model = JetCNN().to(device)
	
	# Prevents Logit Explosion
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
	criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
	
	# Dynamically drops LR when validation loss plateaus
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

	history = {'train_loss': [], 'val_loss': []}

	print(f"\n🚀 Training Optimized CNN for {EPOCHS} epochs...")
	script_start_time = time.time()
	
	for epoch in range(EPOCHS):
		epoch_start_time = time.time()
		model.train()
		total_loss = 0
		
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
		for img, y in pbar:
			img, y = img.to(device, non_blocking=True), y.to(device, non_blocking=True)
			
			# --- PHYSICS DATA AUGMENTATION ---
			# Randomly flip eta (dim 2) and phi (dim 3) 50% of the time
			if torch.rand(1).item() > 0.5:
				img = torch.flip(img, dims=[2])
			if torch.rand(1).item() > 0.5:
				img = torch.flip(img, dims=[3])
			
			optimizer.zero_grad(set_to_none=True)
			loss = criterion(model(img), y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		
		model.eval()
		v_loss = 0
		with torch.no_grad():
			val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
			for img, y in val_pbar:
				img, y = img.to(device, non_blocking=True), y.to(device, non_blocking=True)
				v_loss += criterion(model(img), y).item()
				
		avg_train = total_loss / len(train_loader)
		avg_val = v_loss / len(val_loader)
		history['train_loss'].append(avg_train)
		history['val_loss'].append(avg_val)
		
		# Step the scheduler based on validation performance
		scheduler.step(avg_val)
		
		epoch_time = time.time() - epoch_start_time
		formatted_time = str(datetime.timedelta(seconds=int(epoch_time)))
		
		print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {formatted_time}")

	np.save(os.path.join(RESULTS_DIR, "loss_history.npy"), history)

	# --- Final Evaluation ---
	print("\nExtracting scores for Train, Val, and Test datasets...")
	train_y, train_probs = get_model_scores(model, eval_train_loader, device, desc="Eval Train")
	val_y, val_probs = get_model_scores(model, val_loader, device, desc="Eval Val")
	test_y, test_probs = get_model_scores(model, test_loader, device, desc="Eval Test")

	fpr_t, tpr_t, thresholds = roc_curve(test_y, test_probs)
	opt_threshold = thresholds[np.argmax(tpr_t - fpr_t)]
	
	np.savez_compressed(
		os.path.join(RESULTS_DIR, "cnn_scores_optimized.npz"),
		train_y=train_y, train_probs=train_probs,
		val_y=val_y, val_probs=val_probs,
		test_y=test_y, test_probs=test_probs
	)

	# ==========================================
	# 5. Publication-Ready Visuals
	# ==========================================
	print("\nGenerating and saving plots...")
	plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

	# 1. Learning Curve
	plt.figure(figsize=(8, 6), dpi=300)
	epochs_range = range(1, EPOCHS + 1)
	plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue', lw=2)
	plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='red', lw=2)
	plt.xlabel('Epoch', weight='bold')
	plt.ylabel('BCE Loss', weight='bold')
	plt.title('CNN Learning Curve', weight='bold')
	plt.legend(frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 2. Combined ROC Curve
	plt.figure(figsize=(8, 6), dpi=300)
	fpr_tr, tpr_tr, _ = roc_curve(train_y, train_probs)
	plt.plot(fpr_tr, tpr_tr, color='blue', lw=2, linestyle=':', label=f'Train (AUC = {auc(fpr_tr, tpr_tr):.4f})')
	fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
	plt.plot(fpr_v, tpr_v, color='green', lw=2, linestyle='-.', label=f'Val (AUC = {auc(fpr_v, tpr_v):.4f})')
	plt.plot(fpr_t, tpr_t, color='darkorange', lw=2.5, label=f'Test (AUC = {auc(fpr_t, tpr_t):.4f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate (QCD Mistag)', weight='bold')
	plt.ylabel('True Positive Rate (Top Efficiency)', weight='bold')
	plt.title('CNN Combined ROC Curves', weight='bold')
	plt.legend(loc="lower right", frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/roc_curve_combined.png", dpi=300, bbox_inches='tight')
	plt.close()

	# 3. Confusion Matrix
	cm = confusion_matrix(test_y, test_probs >= opt_threshold)
	plt.figure(figsize=(8, 6), dpi=300)
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'], annot_kws={"size": 16})
	plt.title(f'CNN Confusion Matrix (Threshold = {opt_threshold:.3f})', weight='bold')
	plt.xlabel('Predicted Label', weight='bold')
	plt.ylabel('True Label', weight='bold')
	plt.tight_layout()
	plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
	plt.close()
	
	total_time = time.time() - script_start_time
	print(f"✅ Complete. Total Time: {str(datetime.timedelta(seconds=int(total_time)))}")
	print(f"📁 Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
	run_training()
