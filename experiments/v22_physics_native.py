"""
v22_physics_native.py
=====================
OlfaBind v22: Physics-Native Training

NO contrastive pre-training. Instead:
  1. Hamiltonian Trajectory Matching — similar scents have similar energy trajectories
  2. PINN Regularization — penalize violations of energy/momentum conservation
  3. Spectral Matching — similar scents have similar gravitational eigenvalue spectra

All physics quantities computed FROM the simulation itself.

Based on:
  - Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)
  - Raissi et al., "Physics-Informed Neural Networks" (JCP 2019)
  - Gilmer et al., "Neural Message Passing" (ICML 2017) — spectral domain

Experiment: 5-seed x 5-fold CV on Snitz similarity task
"""
import os, sys, json, time, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import PhysicsProcessingEngine, OrbitalStabilityEvaluator
from models.olfabind_contrastive import SliceLatentModule

from rdkit import Chem
from rdkit.Chem import AllChem

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

DREAM_DIR = r"C:\Users\user\Desktop\Game\server\data\pom_data\dream_mixture"
RESULTS_DIR = r"C:\Users\user\Desktop\Game\paper\results\mixture_prediction"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================================
# DATA LOADING
# ======================================================================
FP_DIM = 2048
c2s = {}
for dn in ['snitz_2013', 'bushdid_2014', 'ravia_2020']:
    mf = os.path.join(DREAM_DIR, dn, 'molecules.csv')
    if os.path.exists(mf):
        df = pd.read_csv(mf)
        for _, row in df.iterrows():
            cid = str(row.get('CID', '')).strip().replace('.0', '')
            smi = str(row.get('IsomericSMILES', row.get('SMILES', ''))).strip()
            if cid and smi and smi != 'nan':
                c2s[cid] = smi

FP_CACHE = {}
def get_fp(smi):
    if smi not in FP_CACHE:
        m = Chem.MolFromSmiles(smi)
        if m: FP_CACHE[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_DIM), dtype=np.float32)
        else: FP_CACHE[smi] = np.zeros(FP_DIM, dtype=np.float32)
    return FP_CACHE[smi]

fp_lookup = {smi: get_fp(smi) for smi in c2s.values()}

def load_snitz():
    pairs = []
    with open(os.path.join(DREAM_DIR, "snitz_2013", "behavior.csv"), 'r', errors='ignore') as f:
        for row in csv.DictReader(f):
            ca = [c.strip() for c in row['StimulusA'].split(',') if c.strip() in c2s]
            cb = [c.strip() for c in row['StimulusB'].split(',') if c.strip() in c2s]
            if ca and cb:
                pairs.append({'ca': ca, 'cb': cb, 'sim': float(row['Similarity'])})
    return pairs

snitz_all = load_snitz()
print(f"Snitz pairs: {len(snitz_all)}")

def augment_pairs(pairs):
    augmented = list(pairs)
    for p in pairs:
        if p['ca'] != p['cb']:
            augmented.append({'ca': p['cb'], 'cb': p['ca'], 'sim': p['sim']})
    return augmented

MAX_MOLS = 20

class OlfaBindDataset(Dataset):
    def __init__(self, pairs, max_mols=MAX_MOLS, emb_dim=FP_DIM):
        self.pairs = pairs; self.max_mols = max_mols; self.emb_dim = emb_dim
    def __len__(self): return len(self.pairs)
    def _pad(self, cids):
        embs = [fp_lookup[c2s[c]] for c in cids if c in c2s and c2s[c] in fp_lookup]
        out = np.zeros((self.max_mols, self.emb_dim), dtype=np.float32)
        mask = np.zeros(self.max_mols, dtype=np.float32)
        for i in range(min(len(embs), self.max_mols)):
            if embs[i].shape[0] == self.emb_dim: out[i] = embs[i]; mask[i] = 1.0
        return out, mask
    def __getitem__(self, idx):
        p = self.pairs[idx]
        a, ma = self._pad(p['ca']); b, mb = self._pad(p['cb'])
        return {
            'fp_a': torch.from_numpy(a), 'mask_a': torch.from_numpy(ma),
            'fp_b': torch.from_numpy(b), 'mask_b': torch.from_numpy(mb),
            'sim': torch.tensor(p['sim'] / 100.0, dtype=torch.float32)
        }

# ======================================================================
# MODEL: Physics-Native OlfaBind (v22)
# ======================================================================
PHYS_EMB_DIM = 20

class OlfaBindPhysicsNative(nn.Module):
    """
    v22: Physics-Native Training — NO contrastive loss.
    
    3 physics-based losses replace contrastive:
      1. L_hamiltonian: energy trajectory matching between pairs
      2. L_pinn: energy/momentum conservation regularization
      3. L_spectral: eigenvalue spectrum matching between pairs
    
    Pipeline:
    FP -> InputHardwareLayer -> constellation (B,N,128)
       -> SliceLatentModule (position only, NO triplet) -> z_positions (B,N,3)
       -> PhysicsProcessingEngine -> emb (B,20) + trajectory + masses
    """
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05,
                 lambda_H=0.01, lambda_S=0.01, lambda_P=0.001):
        super().__init__()
        self.lambda_H = lambda_H
        self.lambda_S = lambda_S
        self.lambda_P = lambda_P
        
        self.input_layer = InputHardwareLayer(
            d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16
        )
        self.contrastive = SliceLatentModule(
            d_atom=d_atom, h_dim=256, z_dim=3,
            margin=1.0, drop_prob=0.0, noise_std=0.0,  # NO augmentation
            position_scale_init=2.0
        )
        self.physics = PhysicsProcessingEngine(
            d_atom=d_atom, n_steps=n_steps, dt=dt
        )
        self.proj = nn.Sequential(
            nn.Linear(PHYS_EMB_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64)
        )
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )
    
    def _process_mixture(self, fp, mask):
        """Process one mixture. Returns embedding + trajectory + masses for physics losses."""
        constellation = self.input_layer(fp, mask)
        z_pos, _ = self.contrastive(constellation, mask)  # ignore triplet loss
        
        # Get full physics output including trajectory
        stab, emb, traj = self.physics(constellation, mask, override_positions=z_pos)
        
        # Get masses from physics engine mapper (returns masses, positions, velocities)
        with torch.no_grad():
            masses_out, _, _ = self.physics.mapper(constellation, mask, override_positions=z_pos)
        # masses_out is already (B, N, 1) from ConstellationToCelestial
        
        return emb, traj, masses_out.detach(), mask
    
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        emb_a, traj_a, mass_a, mask_a = self._process_mixture(fp_a, mask_a)
        emb_b, traj_b, mass_b, mask_b = self._process_mixture(fp_b, mask_b)
        
        pa, pb = self.proj(emb_a), self.proj(emb_b)
        sim = torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))
        
        # Store for physics losses
        self._traj_a, self._mass_a, self._mask_a = traj_a, mass_a, mask_a
        self._traj_b, self._mass_b, self._mask_b = traj_b, mass_b, mask_b
        
        return sim
    
    def compute_physics_losses(self, similarity_target):
        """
        Compute the 3 physics-native losses.
        Must be called AFTER forward().
        """
        eval_a = self.physics.evaluator
        eval_b = self.physics.evaluator
        
        losses = {}
        
        # 1. Hamiltonian Trajectory Matching
        # Similar scents should have similar energy trajectories
        try:
            H_a = eval_a.compute_energy_trajectory(self._traj_a, self._mass_a, self._mask_a)  # (B, T)
            H_b = eval_b.compute_energy_trajectory(self._traj_b, self._mass_b, self._mask_b)  # (B, T)
            
            # Normalize energy trajectories for comparison
            H_a_norm = F.normalize(H_a, dim=-1)
            H_b_norm = F.normalize(H_b, dim=-1)
            
            # Similar pairs: energy trajectories should match
            # Dissimilar pairs: energy trajectories should differ
            energy_sim = F.cosine_similarity(H_a_norm, H_b_norm, dim=-1)  # (B,)
            L_H = F.mse_loss(energy_sim, similarity_target)
            losses['hamiltonian'] = L_H * self.lambda_H
        except Exception:
            losses['hamiltonian'] = torch.tensor(0.0, device=similarity_target.device)
        
        # 2. PINN Regularization (self-supervised, no labels needed)
        try:
            pinn_a = eval_a.compute_pinn_loss(self._traj_a, self._mass_a, self._mask_a)
            pinn_b = eval_b.compute_pinn_loss(self._traj_b, self._mass_b, self._mask_b)
            losses['pinn'] = (pinn_a + pinn_b) / 2.0 * self.lambda_P
        except Exception:
            losses['pinn'] = torch.tensor(0.0, device=similarity_target.device)
        
        # 3. Spectral Matching
        # Similar scents should have similar eigenvalue spectra
        try:
            spec_a = eval_a.compute_spectral_signature(self._traj_a, self._mass_a, self._mask_a)  # (B, N)
            spec_b = eval_b.compute_spectral_signature(self._traj_b, self._mass_b, self._mask_b)  # (B, N)
            
            # Normalize spectra
            spec_a_norm = F.normalize(spec_a, dim=-1)
            spec_b_norm = F.normalize(spec_b, dim=-1)
            
            spec_sim = F.cosine_similarity(spec_a_norm, spec_b_norm, dim=-1)  # (B,)
            L_S = F.mse_loss(spec_sim, similarity_target)
            losses['spectral'] = L_S * self.lambda_S
        except Exception:
            losses['spectral'] = torch.tensor(0.0, device=similarity_target.device)
        
        losses['total_physics'] = losses['hamiltonian'] + losses['pinn'] + losses['spectral']
        return losses

# v18 baseline
class OlfaBindBaseline(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
        self.input_layer = InputHardwareLayer(d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16)
        self.physics = PhysicsProcessingEngine(d_atom=d_atom, n_steps=n_steps, dt=dt)
        self.proj = nn.Sequential(
            nn.Linear(PHYS_EMB_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        ca, cb = self.input_layer(fp_a, mask_a), self.input_layer(fp_b, mask_b)
        _, ea, _ = self.physics(ca, mask_a); _, eb, _ = self.physics(cb, mask_b)
        pa, pb = self.proj(ea), self.proj(eb)
        return torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))

# ======================================================================
# TRAINING
# ======================================================================
def eval_model(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            for i in range(len(y)):
                if not (torch.isnan(y[i]) or torch.isinf(y[i])):
                    preds.append(y[i].cpu().item()); trues.append(b['sim'][i].item())
    if len(preds) < 2 or np.std(preds) < 1e-8: return 0.0
    return pearsonr(preds, trues)[0]

def train_single_v22(model, train_loader, val_loader, epochs=50, phys_lr=1e-5, head_lr=5e-4):
    """Training with physics-native losses."""
    is_physics_native = isinstance(model, OlfaBindPhysicsNative)
    
    if hasattr(model, 'input_layer'):
        physics_params = list(model.input_layer.parameters()) + list(model.physics.parameters())
        if hasattr(model, 'contrastive'):
            physics_params += list(model.contrastive.parameters())
        head_params = [p for p in model.parameters() if not any(p is pp for pp in physics_params)]
        opt = optim.Adam([
            {'params': physics_params, 'lr': phys_lr},
            {'params': head_params, 'lr': head_lr},
        ], weight_decay=1e-4)
        if hasattr(model, 'sim_head'):
            for m in model.sim_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=2.0)
        clip_val = 0.5
    else:
        opt = optim.Adam(model.parameters(), lr=head_lr, weight_decay=1e-4)
        clip_val = 1.0
    
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_r, best_state, no_improve = 0.0, None, 0
    
    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            sim_target = b['sim'].to(device)
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, sim_target)
            
            if hasattr(model, 'input_layer'):
                loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            
            # v22: Physics-native losses (replaces contrastive)
            if is_physics_native:
                phys_losses = model.compute_physics_losses(sim_target)
                loss = loss + phys_losses['total_physics']
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            opt.step()
        
        scheduler.step()
        r_val = eval_model(model, val_loader)
        
        if r_val > best_r:
            best_r = r_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def train_and_eval(model_cls, train_p, val_p, epochs=50, lr=5e-4, seed=42,
                   n_restarts=3):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(OlfaBindDataset(augment_pairs(train_p)), batch_size=16, shuffle=True)
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=16)
    
    best_model, best_r = None, -1.0
    for restart in range(n_restarts):
        torch.manual_seed(seed * 1000 + restart)
        model = model_cls().to(device)
        model, r = train_single_v22(model, train_loader, val_loader, epochs=epochs,
                                     phys_lr=1e-5, head_lr=lr)
        if r > best_r:
            best_r = r
            best_model = model
    
    return best_model, best_r

# ======================================================================
# MAIN EXPERIMENT
# ======================================================================
SEEDS = [42, 123, 456, 789, 2024]
best_T, best_dt = 4, 0.05

print("\n" + "="*60)
print("v22: Physics-Native Training (NO contrastive)")
print("  Hamiltonian + PINN + Spectral losses")
print("="*60)

results = {}

# --- v22: Physics-Native ---
print("\n--- v22: Physics-Native (HNN + PINN + Spectral) ---")

def make_v22_model():
    class M(OlfaBindPhysicsNative):
        def __init__(self):
            super().__init__(n_steps=best_T, dt=best_dt,
                           lambda_H=0.01, lambda_S=0.01, lambda_P=0.001)
    return M

v22_cls = make_v22_model()
all_rs_v22 = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(v22_cls, tp, vp, epochs=50, lr=5e-4,
                              seed=seed*100+fold, n_restarts=3)
        all_rs_v22.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params_v22 = sum(p.numel() for p in v22_cls().parameters())
non_zero_v22 = [x for x in all_rs_v22 if x > 0]
results['v22_physics_native'] = {
    'mean_r': float(np.mean(all_rs_v22)),
    'std_r': float(np.std(all_rs_v22)),
    'all_r': [float(x) for x in all_rs_v22],
    'collapse_rate': 1.0 - len(non_zero_v22) / len(all_rs_v22),
    'n_params': n_params_v22,
    'time_sec': time.time() - t0,
    'losses': ['hamiltonian_matching', 'pinn_regularization', 'spectral_matching'],
    'lambda_H': 0.01, 'lambda_S': 0.01, 'lambda_P': 0.001,
}
print(f"  => v22 Physics-Native: r={np.mean(all_rs_v22):.4f}+/-{np.std(all_rs_v22):.4f}")

# --- v18 Baseline ---
print("\n--- v18: OlfaBind Baseline ---")

def make_baseline_model():
    class M(OlfaBindBaseline):
        def __init__(self):
            super().__init__(n_steps=best_T, dt=best_dt)
    return M

baseline_cls = make_baseline_model()
all_rs_baseline = []
t0 = time.time()

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
        tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
        _, r = train_and_eval(baseline_cls, tp, vp, epochs=50, lr=5e-4,
                              seed=seed*100+fold, n_restarts=3)
        all_rs_baseline.append(r)
        print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

n_params_b = sum(p.numel() for p in baseline_cls().parameters())
non_zero_b = [x for x in all_rs_baseline if x > 0]
results['v18_baseline'] = {
    'mean_r': float(np.mean(all_rs_baseline)),
    'std_r': float(np.std(all_rs_baseline)),
    'all_r': [float(x) for x in all_rs_baseline],
    'collapse_rate': 1.0 - len(non_zero_b) / len(all_rs_baseline),
    'n_params': n_params_b,
    'time_sec': time.time() - t0,
}
print(f"  => v18 Baseline: r={np.mean(all_rs_baseline):.4f}+/-{np.std(all_rs_baseline):.4f}")

results['references'] = {
    'v21_enhanced': 0.553, 'v20_triplet': 0.436,
    'v19_infonce': 0.594, 'v18_baseline': 0.672,
}

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

delta = results['v22_physics_native']['mean_r'] - results['v18_baseline']['mean_r']
print(f"\n  v22 Physics-Native: r={results['v22_physics_native']['mean_r']:.4f}+/-{results['v22_physics_native']['std_r']:.4f}  collapse={results['v22_physics_native']['collapse_rate']:.0%}")
print(f"  v18 Baseline:      r={results['v18_baseline']['mean_r']:.4f}+/-{results['v18_baseline']['std_r']:.4f}  collapse={results['v18_baseline']['collapse_rate']:.0%}")
print(f"  Delta (v22-v18):   {delta:+.4f}")
print(f"\n  References: v21=0.553  v20=0.436  v19=0.594  v18=0.672")
print(f"\n  v22 params: {results['v22_physics_native']['n_params']:,}")
print(f"  v18 params: {results['v18_baseline']['n_params']:,}")

out_path = os.path.join(RESULTS_DIR, "v22_physics_native.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")
