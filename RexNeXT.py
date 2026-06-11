import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnext50_32x4d
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
DATA_FILE = "/home/sachit/ML/processed_data/jetclass_1M/rexnext_dataset_1M.h5"
OUTPUT_DIR = "/home/sachit/ML/jetclass_Results_1M/rexnext"
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-4

# ==========================================
# Dataset (Lazy Loading for Multiprocessing)
# ==========================================
class JetImageDataset(Dataset):
	def __init__(self, h5_path):
		self.h5_path = h5_path
		self.file = None
		
		# Briefly open just to get the total length, then close immediately
		with h5py.File(self.h5_path, 'r') as f:
			self.length = f['images'].shape[0]

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		# Open file lazily on first access per worker process
		if self.file is None:
			self.file = h5py.File(self.h5_path, 'r')
			
		image = self.file['images'][idx]
		label = self.file['labels'][idx]
		
		# 1. Handle missing channel dimension (e.g., 64x64 -> 1x64x64)
		if len(image.shape) == 2:
			image = np.expand_dims(image, axis=0)
			
		# 2. Handle channels-last format if applicable (e.g., 64x64x1 -> 1x64x64)
		elif image.shape[-1] == 1 or image.shape[-1] == 3:
			image = np.transpose(image, (2, 0, 1))
			
		# 3. ResNeXt expects 3 channels. If data is 1 channel, repeat it to RGB.
		if image.shape[0] == 1:
			image = np.repeat(image, 3, axis=0)
			
		# Convert to torch tensors
		image = torch.tensor(image, dtype=torch.float32)
		label = torch.tensor(label, dtype=torch.long)
		
		return image, label

# ==========================================
# Visualization Helpers
# ==========================================
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
	epochs = range(1, len(train_losses) + 1)
	
	plt.figure(figsize=(12, 5))
	
	# Loss Plot
	plt.subplot(1, 2, 1)
	plt.plot(epochs, train_losses, 'b-', label='Training Loss')
	plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	
	# Accuracy Plot
	plt.subplot(1, 2, 2)
	plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
	plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
	plt.savefig(os.path.join(output_dir, 'learning_curves.pdf'))
	plt.close()

# ==========================================
# Main Execution
# ==========================================
def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	
	# --- Device Setup ---
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	if torch.cuda.is_available():
		print(f"GPU Name: {torch.cuda.get_device_name(0)}")

	# --- Dataset & DataLoaders ---
	print(f"Loading dataset from: {DATA_FILE}")
	full_dataset = JetImageDataset(DATA_FILE)
	total_size = len(full_dataset)
	
	train_size = int(0.7 * total_size)
	val_size = int(0.15 * total_size)
	test_size = total_size - train_size - val_size
	
	train_dataset, val_dataset, test_dataset = random_split(
		full_dataset, [train_size, val_size, test_size]
	)
	
	print(f"Splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
	
	# num_workers=4 speeds up HDF5 reading
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

	# --- Model Setup ---
	print("Initializing ResNeXt-50 model...")
	model = resnext50_32x4d(weights=None)
	# Modify final layer for binary classification (QCD vs Top)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 2)
	model = model.to(device)

	# --- Loss & Optimizer ---
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

	# --- Training Loop ---
	best_val_loss = float('inf')
	train_losses, val_losses = [], []
	train_accs, val_accs = [], []

	print("Starting training...")
	for epoch in range(EPOCHS):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		
		# --- Training Progress Bar ---
		train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
		for images, labels in train_pbar:
			images, labels = images.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item() * images.size(0)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			
			# Update progress bar with live metrics
			train_pbar.set_postfix(loss=f"{(running_loss/total):.4f}", acc=f"{(correct/total):.4f}")
			
		epoch_train_loss = running_loss / total
		epoch_train_acc = correct / total
		train_losses.append(epoch_train_loss)
		train_accs.append(epoch_train_acc)
		
		# --- Validation Progress Bar ---
		model.eval()
		val_loss = 0.0
		val_correct = 0
		val_total = 0
		
		val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
		with torch.no_grad():
			for images, labels in val_pbar:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				
				val_loss += loss.item() * images.size(0)
				_, predicted = torch.max(outputs.data, 1)
				val_total += labels.size(0)
				val_correct += (predicted == labels).sum().item()
				
				val_pbar.set_postfix(loss=f"{(val_loss/val_total):.4f}", acc=f"{(val_correct/val_total):.4f}")
				
		epoch_val_loss = val_loss / val_total
		epoch_val_acc = val_correct / val_total
		val_losses.append(epoch_val_loss)
		val_accs.append(epoch_val_acc)
		
		# Print final epoch summary
		print(f"Epoch {epoch+1}/{EPOCHS} | "
			  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
			  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
		
		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_resnext.pth'))
			print("--> Saved new best model")
			
	print("\nGenerating Training Curves...")
	plot_learning_curves(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)
	
	print("Evaluating on Train, Val, and Test Sets...")
	model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_resnext.pth')))
	model.eval()
	
	def get_predictions(loader, desc):
		probs = []
		true_labels = []
		pbar = tqdm(loader, desc=desc)
		with torch.no_grad():
			for images, labels in pbar:
				images = images.to(device)
				outputs = model(images)
				batch_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
				probs.extend(batch_probs)
				true_labels.extend(labels.numpy())
		return np.array(true_labels), np.array(probs)

	# Get predictions for all sets
	train_y, train_probs = get_predictions(train_loader, "Evaluating Train")
	val_y, val_probs = get_predictions(val_loader, "Evaluating Val")
	test_y, test_probs = get_predictions(test_loader, "Evaluating Test")
	
	# --- Save Raw Data ---
	scores_file = os.path.join(OUTPUT_DIR, "resnext_predictions.npz")
	np.savez_compressed(
		scores_file,
		train_y=train_y, train_probs=train_probs,
		val_y=val_y, val_probs=val_probs,
		test_y=test_y, test_probs=test_probs
	)
	print(f"Raw scores saved successfully to: {scores_file}")

	print("\nGenerating Plots...")
	
	# ==========================================
	# CONFIGURATION
	# ==========================================
	plt.rcParams.update({
		"font.family": "sans-serif",
		"font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
		"font.size": 14,
		"axes.labelsize": 16,
		"axes.titlesize": 18,
		"xtick.labelsize": 14,
		"ytick.labelsize": 14,
		"xtick.direction": "in",
		"ytick.direction": "in",
		"xtick.top": True,
		"ytick.right": True,
		"xtick.major.size": 6,
		"ytick.major.size": 6,
		"xtick.minor.size": 3,
		"ytick.minor.size": 3,
		"axes.linewidth": 1.5,
		"legend.frameon": False,
		"legend.fontsize": 13
	})
	
	# --- 1. Confusion Matrix (Test Set) ---
	cm = confusion_matrix(test_y, test_probs > 0.5)
	plt.figure(figsize=(7, 6))
	
	cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	annot = np.empty_like(cm).astype(str)
	for i in range(2):
		for j in range(2):
			annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)"
			
	sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
				xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'],
				annot_kws={"size": 14})
	plt.title('ResNeXt Confusion Matrix (Test Set)', weight='bold')
	plt.ylabel('True Label', weight='bold')
	plt.xlabel('Predicted Label', weight='bold')
	plt.yticks(rotation=0)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_test.pdf"), format='pdf', bbox_inches='tight')
	plt.close()
	
	# --- 2. ROC Curve (Train vs Val vs Test) ---
	plt.figure(figsize=(8, 6))
	
	# Train ROC
	fpr_tr, tpr_tr, _ = roc_curve(train_y, train_probs)
	plt.plot(fpr_tr, tpr_tr, color='blue', lw=2, linestyle=':', label=f'Train (AUC = {auc(fpr_tr, tpr_tr):.4f})')
	
	# Val ROC
	fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
	plt.plot(fpr_v, tpr_v, color='green', lw=2, linestyle='-.', label=f'Val (AUC = {auc(fpr_v, tpr_v):.4f})')
	
	# Test ROC
	fpr_t, tpr_t, _ = roc_curve(test_y, test_probs)
	plt.plot(fpr_t, tpr_t, color='darkorange', lw=2.5, label=f'Test (AUC = {auc(fpr_t, tpr_t):.4f})')
	
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate (QCD Mistag)', weight='bold')
	plt.ylabel('True Positive Rate (Top Efficiency)', weight='bold')
	plt.title('ROC Curves (Overtraining Check)', weight='bold')
	plt.legend(loc="lower right", frameon=True, edgecolor='black')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_comparison.pdf"), format='pdf', bbox_inches='tight')
	plt.close()

	# --- 3. Score Distribution (Test Set) ---
	plt.figure(figsize=(8, 6))
	sns.histplot(test_probs[test_y == 0], bins=50, color='blue', alpha=0.5, label='QCD (Background)', stat='density', element='step', fill=True)
	sns.histplot(test_probs[test_y == 1], bins=50, color='red', alpha=0.5, label='Top (Signal)', stat='density', element='step', fill=True)
	
	plt.hist(test_probs[test_y == 0], bins=50, density=True, color='blue', histtype='step', lw=1.5, alpha=1.0)
	plt.hist(test_probs[test_y == 1], bins=50, density=True, color='red', histtype='step', lw=1.5, alpha=1.0)
	
	plt.xlabel('ResNeXt Score', weight='bold')
	plt.ylabel('Density', weight='bold')
	plt.title('Classifier Score Distribution (Test Set)', weight='bold')
	plt.legend(loc='upper center')
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution_test.png"), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution_test.pdf"), format='pdf', bbox_inches='tight')
	plt.close()
	
	print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
	main()
