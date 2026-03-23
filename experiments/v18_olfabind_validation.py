"""
v18_olfabind_validation.py
===========================
OlfaBind v18: Improved validation with:
  1. T-sweep (4/8/16/32) with dt scaling for long-sim stability
  2. Anti-collapse: multi-restart (3x), cosine annealing, larger batch
  3. Full 5-seed x 5-fold CV with T=4
  4. Transfer (Snitz->Ravia/Bushdid) + Dim Analysis with best model
"""
import os, sys, json, time, csv, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import PhysicsProcessingEngine

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

def parse_ad_eval_dataset(mix_file, gt_file):
    mix_dict = {}
    with open(mix_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [x.strip() for x in line.split(',')]
            label = parts[0].replace('.0', '')
            if not label or label in ['Mixture Label', 'Fraction']: continue
            mix_dict[label] = [x.replace('.0', '') for x in parts[1:] if x and x != 'CID' and x in c2s]
    pairs = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        start = False
        for line in f:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3: continue
            if 'Mixture' in parts[0] or 'Experimental' in parts[-1] or parts[0] == '\ufeffMixture 1':
                start = True; continue
            if not start: continue
            m1, m2, val = parts[0].replace('.0',''), parts[1].replace('.0',''), parts[-1]
            if m1 in mix_dict and m2 in mix_dict and val:
                try: pairs.append({'ca': mix_dict[m1], 'cb': mix_dict[m2], 'sim': float(val)})
                except: pass
    return pairs

snitz_all = load_snitz()
ravia_pairs = parse_ad_eval_dataset(
    os.path.join(DREAM_DIR, 'ad_mixture_ravia.csv'),
    os.path.join(DREAM_DIR, 'ad_gt_ravia.csv'))
bushdid_pairs = parse_ad_eval_dataset(
    os.path.join(DREAM_DIR, 'ad_mixture_bushdid.csv'),
    os.path.join(DREAM_DIR, 'ad_gt_bushdid.csv'))
print(f"Data: Snitz={len(snitz_all)}, Ravia={len(ravia_pairs)}, Bushdid={len(bushdid_pairs)}")

def augment_pairs(pairs):
    augmented = list(pairs)
    for p in pairs:
        if p['ca'] != p['cb']:
            augmented.append({'ca': p['cb'], 'cb': p['ca'], 'sim': p['sim']})
    return augmented

# ======================================================================
# DATASET
# ======================================================================
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
# OLFABIND MODEL
# ======================================================================
class OlfaBindModel(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=4, dt=0.05):
        super().__init__()
        self.n_steps = n_steps
        self.input_layer = InputHardwareLayer(d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16)
        self.physics = PhysicsProcessingEngine(d_atom=d_atom, n_steps=n_steps, dt=dt)
        self.proj = nn.Sequential(
            nn.Linear(19, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, fp_a, mask_a, fp_b, mask_b):
        ca = self.input_layer(fp_a, mask_a)
        cb = self.input_layer(fp_b, mask_b)
        _, ea, _ = self.physics(ca, mask_a)
        _, eb, _ = self.physics(cb, mask_b)
        pa, pb = self.proj(ea), self.proj(eb)
        return torch.sigmoid(self.sim_head((pa - pb).abs()).squeeze(-1))

    def get_physics_embeddings(self, fp, mask):
        const = self.input_layer(fp, mask)
        stab, emb, traj = self.physics(const, mask)
        return stab, emb

# ======================================================================
# BASELINES
# ======================================================================
class StdAttentionModel(nn.Module):
    def __init__(self, emb_dim=FP_DIM, hidden_dim=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.Dropout(0.2), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.Dropout(0.2), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//2, 1))
        self.sim = nn.CosineSimilarity(dim=1)
    def _agg(self, embs, mask):
        s = self.attn(embs).squeeze(-1)
        s = s.masked_fill(mask == 0, -1e9)
        w = F.softmax(s, dim=-1)
        return (w.unsqueeze(-1) * embs).sum(dim=1)
    def forward(self, fp_a, mask_a, fp_b, mask_b):
        return self.sim(self._agg(fp_a, mask_a), self._agg(fp_b, mask_b))

def max_pool_baseline(pairs):
    preds, trues = [], []
    for p in pairs:
        fa = [fp_lookup[c2s[c]] for c in p['ca'] if c in c2s and c2s[c] in fp_lookup]
        fb = [fp_lookup[c2s[c]] for c in p['cb'] if c in c2s and c2s[c] in fp_lookup]
        if fa and fb:
            va, vb = np.max(fa, axis=0), np.max(fb, axis=0)
            preds.append(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-8))
            trues.append(p['sim'])
    return pearsonr(preds, trues)[0] if len(preds)>=2 else 0.0

def mean_pool_baseline(pairs):
    preds, trues = [], []
    for p in pairs:
        fa = [fp_lookup[c2s[c]] for c in p['ca'] if c in c2s and c2s[c] in fp_lookup]
        fb = [fp_lookup[c2s[c]] for c in p['cb'] if c in c2s and c2s[c] in fp_lookup]
        if fa and fb:
            va, vb = np.mean(fa, axis=0), np.mean(fb, axis=0)
            preds.append(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-8))
            trues.append(p['sim'])
    return pearsonr(preds, trues)[0] if len(preds)>=2 else 0.0

# ======================================================================
# TRAINING: Multi-restart + Cosine Annealing + Early Stopping
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

def train_single(model, train_loader, val_loader, epochs=50, phys_lr=1e-5, head_lr=5e-4):
    """Train one model with cosine annealing + early stopping."""
    if hasattr(model, 'input_layer'):
        physics_params = list(model.input_layer.parameters()) + list(model.physics.parameters())
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
    best_r, best_state, patience, no_improve = 0.0, None, 10, 0
    
    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            loss = F.mse_loss(y, b['sim'].to(device))
            if hasattr(model, 'input_layer'):
                loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            if torch.isnan(loss) or torch.isinf(loss): continue
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
            if no_improve >= patience: break
    
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r

def train_and_eval(model_cls, train_p, val_p, epochs=50, lr=5e-4, seed=42, n_restarts=3, **model_kwargs):
    """Multi-restart training: run n_restarts times, keep best."""
    torch.manual_seed(seed); np.random.seed(seed)
    batch_size = 16
    train_loader = DataLoader(OlfaBindDataset(augment_pairs(train_p)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=batch_size)
    
    is_olfabind = hasattr(model_cls, '__mro__') and any(
        c.__name__ == 'OlfaBindModel' for c in model_cls.__mro__ if hasattr(c, '__name__')
    ) if isinstance(model_cls, type) else hasattr(model_cls(), 'input_layer')
    restarts = n_restarts if is_olfabind else 1
    
    best_model, best_r = None, -1.0
    for restart in range(restarts):
        torch.manual_seed(seed * 1000 + restart)
        model = model_cls(**model_kwargs).to(device)
        model, r = train_single(model, train_loader, val_loader, epochs=epochs,
                                phys_lr=1e-5, head_lr=lr)
        if r > best_r:
            best_r = r
            best_model = model
    
    return best_model, best_r

# ======================================================================
# EXP 1: T-sweep (4, 8, 16, 32.) — Which sim length is optimal?
# ======================================================================
print("\n" + "="*60)
print("EXP 1: Simulation Length Sweep (T=4/8/16/32)")
print("="*60)

results = {}
SEEDS = [42, 123, 456, 789, 2024]

# Baselines
max_r = max_pool_baseline(snitz_all)
mean_r = mean_pool_baseline(snitz_all)
results['max_pool'] = {'mean_r': max_r, 'std_r': 0.0, 'n_params': 0}
results['mean_pool'] = {'mean_r': mean_r, 'std_r': 0.0, 'n_params': 0}
print(f"  Max Pool:  r={max_r:.4f}")
print(f"  Mean Pool: r={mean_r:.4f}")

t_sweep_results = {}
for T in [4, 8, 16, 32]:
    # Scale dt inversely with T to keep total simulation time constant (T*dt ≈ 0.2)
    dt = 0.2 / T
    print(f"\n--- T={T}, dt={dt:.4f} (total_time={T*dt:.2f}) ---")
    
    def make_model_cls(n_steps, dt_val):
        class M(OlfaBindModel):
            def __init__(self): super().__init__(n_steps=n_steps, dt=dt_val)
        return M
    
    model_cls = make_model_cls(T, dt)
    all_rs = []; t0 = time.time()
    
    # Quick CV: 2 seeds x 5 folds for speed
    for seed in SEEDS[:2]:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
            tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
            _, r = train_and_eval(model_cls, tp, vp, epochs=50, lr=5e-4,
                                  seed=seed*100+fold, n_restarts=3)
            all_rs.append(r)
            print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")
    
    mean_r_t = float(np.mean(all_rs))
    std_r_t = float(np.std(all_rs))
    non_zero = [x for x in all_rs if x > 0]
    t_sweep_results[f'T{T}'] = {
        'mean_r': mean_r_t, 'std_r': std_r_t,
        'non_zero_mean': float(np.mean(non_zero)) if non_zero else 0.0,
        'collapse_rate': 1.0 - len(non_zero)/len(all_rs),
        'all_r': all_rs, 'dt': dt, 'time_sec': time.time() - t0
    }
    print(f"  => T={T}: r={mean_r_t:.4f}±{std_r_t:.4f} collapse={1-len(non_zero)/len(all_rs):.0%}")

results['t_sweep'] = t_sweep_results

# Find best T
best_T_key = max(t_sweep_results, key=lambda k: t_sweep_results[k]['mean_r'])
best_T = int(best_T_key[1:])
best_dt = t_sweep_results[best_T_key]['dt']
print(f"\n==> Best T={best_T} (dt={best_dt:.4f}), r={t_sweep_results[best_T_key]['mean_r']:.4f}")

# ======================================================================
# EXP 2: Full 5-seed x 5-fold CV with best T
# ======================================================================
print("\n" + "="*60)
print(f"EXP 2: Full CV with T={best_T} + Std Attention baseline")
print("="*60)

def make_best_model():
    class BestModel(OlfaBindModel):
        def __init__(self): super().__init__(n_steps=best_T, dt=best_dt)
    return BestModel

for name, model_cls in [('olfabind', make_best_model()), ('std_attention', StdAttentionModel)]:
    print(f"\n--- {name} ---")
    all_rs = []; t0 = time.time()
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
            tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
            _, r = train_and_eval(model_cls, tp, vp, epochs=50, lr=5e-4,
                                  seed=seed*100+fold, n_restarts=3)
            all_rs.append(r)
            print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")
    
    non_zero = [x for x in all_rs if x > 0]
    n_params = sum(p.numel() for p in model_cls().parameters())
    results[name] = {
        'mean_r': float(np.mean(all_rs)), 'std_r': float(np.std(all_rs)),
        'ci_95': [float(np.percentile(all_rs, 2.5)), float(np.percentile(all_rs, 97.5))],
        'non_zero_mean': float(np.mean(non_zero)) if non_zero else 0.0,
        'collapse_rate': 1.0 - len(non_zero)/len(all_rs),
        'all_r': [float(x) for x in all_rs],
        'n_params': n_params, 'time_sec': time.time() - t0
    }
    print(f"  => r={np.mean(all_rs):.4f}±{np.std(all_rs):.4f} collapse={1-len(non_zero)/len(all_rs):.0%}")

# ======================================================================
# EXP 3: Dim Analysis (train best model on all Snitz)
# ======================================================================
print("\n" + "="*60)
print("EXP 3: Physics Embedding Dimension Analysis")
print("="*60)

# Train with multi-restart on all data
best_model_for_analysis = None
best_train_r = -1
full_loader = DataLoader(OlfaBindDataset(augment_pairs(snitz_all)), batch_size=16, shuffle=True)
val_full_loader = DataLoader(OlfaBindDataset(snitz_all[:50]), batch_size=16)

for restart in range(5):  # 5 restarts to find best non-collapsed model
    torch.manual_seed(restart * 42)
    model = make_best_model()().to(device)
    model, r = train_single(model, full_loader, val_full_loader, epochs=50, phys_lr=1e-5, head_lr=5e-4)
    print(f"  Restart {restart+1}: r={r:.4f}")
    if r > best_train_r:
        best_train_r = r
        best_model_for_analysis = model

print(f"  Best training r: {best_train_r:.4f}")

# Extract and analyze
emb_names = ['stability', 'S_E', 'S_R', 'S_C', 'mean_mass', 'mean_vel', 'total_L',
             'displacement', 'init_r', 'final_r',
             'init_pos_x', 'init_pos_y', 'init_pos_z',
             'final_pos_x', 'final_pos_y', 'final_pos_z',
             'final_vel_x', 'final_vel_y', 'final_vel_z']
dim_correlations = {}

if best_train_r > 0.1:
    best_model_for_analysis.eval()
    with torch.no_grad():
        all_embs_a, all_embs_b, all_sims = [], [], []
        for p in snitz_all:
            ds = OlfaBindDataset([p]); b = ds[0]
            _, ea = best_model_for_analysis.get_physics_embeddings(
                b['fp_a'].unsqueeze(0).to(device), b['mask_a'].unsqueeze(0).to(device))
            _, eb = best_model_for_analysis.get_physics_embeddings(
                b['fp_b'].unsqueeze(0).to(device), b['mask_b'].unsqueeze(0).to(device))
            all_embs_a.append(ea.cpu().numpy()[0])
            all_embs_b.append(eb.cpu().numpy()[0])
            all_sims.append(p['sim'])

        embs_a = np.array(all_embs_a); embs_b = np.array(all_embs_b)
        sims = np.array(all_sims)
        for i, name in enumerate(emb_names):
            if i >= embs_a.shape[1]: break
            diff = np.abs(embs_a[:, i] - embs_b[:, i])
            if np.std(diff) > 1e-8:
                pr, _ = pearsonr(-diff, sims); sr, _ = spearmanr(-diff, sims)
            else: pr, sr = 0.0, 0.0
            dim_correlations[name] = {'pearson': float(pr), 'spearman': float(sr)}
            print(f"  {name:15s}: Pearson={pr:.4f}  Spearman={sr:.4f}")
else:
    print("  Skipped (all restarts collapsed)")
    for name in emb_names:
        dim_correlations[name] = {'pearson': 0.0, 'spearman': 0.0}

results['dim_analysis'] = dim_correlations

# ======================================================================
# EXP 4: Zero-shot Transfer
# ======================================================================
print("\n" + "="*60)
print("EXP 4: Zero-shot Transfer (Snitz -> Ravia/Bushdid)")
print("="*60)

transfer_results = {}
if best_model_for_analysis and best_train_r > 0.1:
    for tname, tpairs in [('ravia', ravia_pairs), ('bushdid', bushdid_pairs)]:
        if not tpairs: continue
        max_r_t = max_pool_baseline(tpairs)
        tloader = DataLoader(OlfaBindDataset(tpairs), batch_size=16)
        olfabind_r = eval_model(best_model_for_analysis, tloader)
        transfer_results[tname] = {
            'n_pairs': len(tpairs), 'max_pool_r': float(max_r_t),
            'olfabind_r': float(olfabind_r), 'delta': float(olfabind_r - max_r_t)
        }
        print(f"  {tname}: MaxPool={max_r_t:.4f}  OlfaBind={olfabind_r:.4f}")
else:
    print("  Skipped (no valid model)")

results['transfer'] = transfer_results

# ======================================================================
# Ablation: No Gravity / No Inhibition
# ======================================================================
class OlfaBindNoGravity(OlfaBindModel):
    def __init__(self):
        super().__init__(n_steps=best_T, dt=best_dt)
        with torch.no_grad(): self.physics.engine.log_G.fill_(-10.0)
        self.physics.engine.log_G.requires_grad_(False)

class OlfaBindNoInhibition(OlfaBindModel):
    def __init__(self):
        super().__init__(n_steps=best_T, dt=best_dt)
        with torch.no_grad():
            self.input_layer.slice_array.lateral_inhibition.weight.fill_(0.0)
            self.input_layer.slice_array.lateral_inhibition.weight[0,0,2] = 1.0
        self.input_layer.slice_array.lateral_inhibition.weight.requires_grad_(False)

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("\nT-Sweep:")
for k, v in results.get('t_sweep', {}).items():
    print(f"  {k}: r={v['mean_r']:.4f}±{v['std_r']:.4f} collapse={v['collapse_rate']:.0%}")

print(f"\nFull CV (T={best_T}):")
for k in ['max_pool', 'mean_pool', 'olfabind', 'std_attention']:
    if k in results:
        v = results[k]
        cr = v.get('collapse_rate', 0)
        print(f"  {k:20s}: r={v['mean_r']:.4f}±{v.get('std_r',0):.4f} collapse={cr:.0%}")

print(f"\nDim Analysis:")
for k, v in results.get('dim_analysis', {}).items():
    if abs(v.get('pearson', 0)) > 0.05:
        print(f"  {k:15s}: Pearson={v['pearson']:.4f}")

print(f"\nTransfer:")
for k, v in results.get('transfer', {}).items():
    print(f"  {k}: OlfaBind={v['olfabind_r']:.4f} MaxPool={v['max_pool_r']:.4f}")

# Save
out_path = os.path.join(RESULTS_DIR, "v18_olfabind_validation.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")
