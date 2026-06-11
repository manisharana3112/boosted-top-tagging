import os
import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from tqdm import tqdm

# ==========================================
# 1. HARDWARE & DIRECTORY CONFIG
# ==========================================
INPUT_H5 = "/home/sachit/ML/processed_data/jetclass_1M/BDT_optimized_dataset.h5"
RESULTS_DIR = "/home/sachit/ML/jetclass_Results_1M/BDT_Max"

DEVICE = "cuda:0" 
N_THREADS = -1 
# ==========================================

class TqdmCallback(xgb.callback.TrainingCallback):
	def __init__(self, total_epochs):
		self.pbar = tqdm(total=total_epochs, desc="🚀 GPU Training", unit="tree")
	def after_iteration(self, model, epoch, evals_log):
		self.pbar.update(1)
		return False
	def after_training(self, model):
		self.pbar.close()
		return model

def set_plot_style():
	sns.set_theme(style="whitegrid")
	plt.rcParams.update({
		'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20,
		'legend.fontsize': 14, 'axes.linewidth': 2.0, 'grid.alpha': 0.7
	})

def run_max_power_training():
	os.makedirs(RESULTS_DIR, exist_ok=True)
	set_plot_style()

	# 1. Data Loading
	if not os.path.exists(INPUT_H5):
		print(f"❌ ERROR: File not found at {INPUT_H5}")
		return

	with h5py.File(INPUT_H5, 'r') as f:
		X = f['features'][:]
		y = f['labels'][:]
		feature_names = f.attrs.get('feature_names', [f"Feat_{i}" for i in range(X.shape[1])])
	
	feature_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in feature_names]
	
	# 2. 60/20/20 Split
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

	# 3. Native DMatrix
	dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
	dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
	dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

	# 4. Modern Hyperparameters for XGBoost 3.x+
	params = {
		'objective': 'binary:logistic',
		'device': DEVICE,
		'tree_method': 'hist',
		'eval_metric': ['logloss', 'auc'],
		'eta': 0.05,
		'max_depth': 7,
		'subsample': 0.8,
		'colsample_bytree': 0.8,
		'nthread': N_THREADS
	}

	evals_result = {}
	n_rounds = 2000

	# 5. Training
	bst = xgb.train(
		params, dtrain, n_rounds,
		evals=[(dtrain, 'train'), (dval, 'val')],
		early_stopping_rounds=100,
		evals_result=evals_result,
		callbacks=[TqdmCallback(n_rounds)],
		verbose_eval=False
	)

	# 6. Inference
	y_prob_train = bst.predict(dtrain)
	y_prob_val = bst.predict(dval)
	y_prob_test = bst.predict(dtest)
	y_pred_test = (y_prob_test > 0.5).astype(int)

	# Save Scores
	np.savez_compressed(
		os.path.join(RESULTS_DIR, "all_results.npz"),
		y_prob_train=y_prob_train, y_train=y_train,
		y_prob_val=y_prob_val, y_val=y_val,
		y_prob_test=y_prob_test, y_test=y_test,
		train_loss=evals_result['train']['logloss'],
		val_loss=evals_result['val']['logloss']
	)

	# ==========================================
	# 7. PUBLICATION PLOTS
	# ==========================================
	
	# --- Plot A: Loss Curves ---
	plt.figure(figsize=(10, 6))
	plt.plot(evals_result['train']['logloss'], label='Train Loss', lw=2)
	plt.plot(evals_result['val']['logloss'], label='Val Loss', lw=2)
	plt.title('Training Evolution', fontweight='bold')
	plt.xlabel('Trees')
	plt.ylabel('LogLoss')
	plt.legend()
	plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300, bbox_inches='tight')

	# --- Plot B: Combined ROC Curve (Overfitting Check) ---
	fpr_tr, tpr_tr, _ = roc_curve(y_train, y_prob_train)
	fpr_v, tpr_v, _ = roc_curve(y_val, y_prob_val)
	fpr_te, tpr_te, _ = roc_curve(y_test, y_prob_test)

	plt.figure(figsize=(8, 8))
	plt.plot(fpr_tr, tpr_tr, color='dodgerblue', lw=2.5, linestyle=':', label=f'Train AUC = {auc(fpr_tr, tpr_tr):.4f}')
	plt.plot(fpr_v, tpr_v, color='forestgreen', lw=2.5, linestyle='-.', label=f'Val AUC = {auc(fpr_v, tpr_v):.4f}')
	plt.plot(fpr_te, tpr_te, color='red', lw=3, linestyle='-', label=f'Test AUC = {auc(fpr_te, tpr_te):.4f}')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.title('Publication ROC Curve', fontweight='bold')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_combined.png"), dpi=300, bbox_inches='tight')

	# --- Plot C: Confusion Matrix (Raw Counts Only) ---
	cm = confusion_matrix(y_test, y_pred_test)
	
	plt.figure(figsize=(10, 8))
	
	# Generate a custom brown colormap
	brown_cmap = sns.light_palette("saddlebrown", as_cmap=True)
	
	# fmt="," automatically adds commas to large numbers (e.g., 15,000)
	sns.heatmap(cm, annot=True, fmt=",", cmap=brown_cmap, 
				xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'],
				annot_kws={'size': 20, 'weight': 'bold'})
				
	plt.title('Confusion Matrix (Raw Counts)', fontweight='bold')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
	
	# --- Plot D: Score Distribution ---
	plt.figure(figsize=(12, 7))
	plt.hist(y_prob_test[y_test == 0], bins=50, density=True, facecolor=to_rgba('blue', 0.4), 
			 edgecolor='blue', lw=2, histtype='stepfilled', label='QCD')
	plt.hist(y_prob_test[y_test == 1], bins=50, density=True, facecolor=to_rgba('red', 0.4), 
			 edgecolor='red', lw=2, histtype='stepfilled', label='Top')
	plt.title('Score Distribution', fontweight='bold')
	plt.legend()
	plt.savefig(os.path.join(RESULTS_DIR, "score_dist.png"), dpi=300, bbox_inches='tight')

	# --- Plot E: Feature Importance ---
	# Using 'gain' which measures the relative contribution of the corresponding feature 
	# to the model calculated by taking each feature's contribution for each tree in the model.
	scores = bst.get_score(importance_type='gain')
	
	if scores:
		importance_df = pd.DataFrame({
			'Feature': list(scores.keys()),
			'Importance': list(scores.values())
		}).sort_values(by='Importance', ascending=False)
		
		plt.figure(figsize=(12, 8))
		sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
		plt.title('XGBoost Feature Importance (Gain)', fontweight='bold')
		plt.xlabel('Relative Importance')
		plt.ylabel('Feature')
		plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')

	bst.save_model(os.path.join(RESULTS_DIR, "max_power_model.json"))
	print(f"✅ Training completed successfully. Results in {RESULTS_DIR}")

if __name__ == "__main__":
	run_max_power_training()
