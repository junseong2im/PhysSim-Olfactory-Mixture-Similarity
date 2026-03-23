"""
v17_olfabind_validation.py
===========================
OlfaBind Validation: Does physics-based scent representation
correlate with human perceptual similarity?

Experiments:
  1. OlfaBind vs Baselines (5-seed x 5-fold CV on Snitz)
  2. Physics embedding dimension analysis
  3. Zero-shot transfer (Snitz -> Ravia/Bushdid)
  4. Ablation study (remove gravity, resonance, etc.)
"""
import os, sys, json, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import pandas as pd

sys.path.insert(0, r"C:\Users\user\Desktop\Game\server")
from models.olfabind_input import InputHardwareLayer
from models.olfabind_engine import PhysicsProcessingEngine, GravitationalEngine, OrbitalStabilityEvaluator, ConstellationToCelestial
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
# DATA LOADING (same as v10/v15/v16)
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
        if m:
            FP_CACHE[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_DIM), dtype=np.float32)
        else:
            FP_CACHE[smi] = np.zeros(FP_DIM, dtype=np.float32)
    return FP_CACHE[smi]

fp_lookup = {smi: get_fp(smi) for smi in c2s.values()}

def load_snitz():
    d = os.path.join(DREAM_DIR, "snitz_2013")
    pairs = []
    with open(os.path.join(d, "behavior.csv"), 'r', errors='ignore') as f:
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
            if not label or label in ['Mixture Label', 'Fraction']:
                continue
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
            m1, m2, val = parts[0].replace('.0', ''), parts[1].replace('.0', ''), parts[-1]
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
MAX_MOLS = 20  # Capped for N-body simulation stability (N^2 interactions)

class OlfaBindDataset(Dataset):
    def __init__(self, pairs, max_mols=MAX_MOLS, emb_dim=FP_DIM):
        self.pairs = pairs
        self.max_mols = max_mols
        self.emb_dim = emb_dim

    def __len__(self): return len(self.pairs)

    def _pad(self, cids):
        embs = [fp_lookup[c2s[c]] for c in cids if c in c2s and c2s[c] in fp_lookup]
        out = np.zeros((self.max_mols, self.emb_dim), dtype=np.float32)
        mask = np.zeros(self.max_mols, dtype=np.float32)
        for i in range(min(len(embs), self.max_mols)):
            if embs[i].shape[0] == self.emb_dim:
                out[i] = embs[i]; mask[i] = 1.0
        return out, mask

    def __getitem__(self, idx):
        p = self.pairs[idx]
        a, ma = self._pad(p['ca']); b, mb = self._pad(p['cb'])
        return {
            'fp_a': torch.from_numpy(a), 'mask_a': torch.from_numpy(ma),
            'fp_b': torch.from_numpy(b), 'mask_b': torch.from_numpy(mb),
            'sim': torch.tensor(p['sim'] / 100.0, dtype=torch.float32)  # normalize 0~100 -> 0~1
        }

# ======================================================================
# OLFABIND MODEL (Full pipeline: Module 1 + Module 2&3)
# ======================================================================
class OlfaBindModel(nn.Module):
    def __init__(self, d_input=FP_DIM, d_atom=128, n_steps=16, dt=0.05):
        super().__init__()
        self.input_layer = InputHardwareLayer(d_input=d_input, d_atom=d_atom, grid_h=8, grid_w=16)
        self.physics = PhysicsProcessingEngine(d_atom=d_atom, n_steps=n_steps, dt=dt)
        # Projection head: 19d physics embedding -> 64d
        self.proj = nn.Sequential(
            nn.Linear(19, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        # Learnable similarity: MLP on |diff| to scalar
        self.sim_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, fp_a, mask_a, fp_b, mask_b):
        const_a = self.input_layer(fp_a, mask_a)
        const_b = self.input_layer(fp_b, mask_b)
        _, emb_a, _ = self.physics(const_a, mask_a)
        _, emb_b, _ = self.physics(const_b, mask_b)
        proj_a = self.proj(emb_a)
        proj_b = self.proj(emb_b)
        # Similarity from |difference|: captures both direction and magnitude
        diff = (proj_a - proj_b).abs()
        sim = torch.sigmoid(self.sim_head(diff).squeeze(-1))  # (B,) in [0, 1]
        return sim

    def get_physics_embeddings(self, fp, mask):
        const = self.input_layer(fp, mask)
        stab, emb, traj = self.physics(const, mask)
        return stab, emb

# ======================================================================
# STANDARD ATTENTION MODEL (Baseline)
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
        scores = self.attn(embs).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = F.softmax(scores, dim=-1)
        return (w.unsqueeze(-1) * embs).sum(dim=1)

    def forward(self, fp_a, mask_a, fp_b, mask_b):
        return self.sim(self._agg(fp_a, mask_a), self._agg(fp_b, mask_b))

# ======================================================================
# TRAINING & EVALUATION
# ======================================================================
def pearson_loss(preds, targets):
    vp = preds - preds.mean(); vt = targets - targets.mean()
    return 1.0 - (vp*vt).mean() / (torch.sqrt((vp**2).mean()+1e-8) * torch.sqrt((vt**2).mean()+1e-8))

def eval_model(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            # Filter NaN predictions
            for i in range(len(y)):
                if not (torch.isnan(y[i]) or torch.isinf(y[i])):
                    preds.append(y[i].cpu().item()); trues.append(b['sim'][i].item())
    if len(preds) < 2 or np.std(preds) < 1e-8:
        return 0.0
    return pearsonr(preds, trues)[0]

def train_and_eval(model_cls, train_p, val_p, epochs=50, lr=5e-4, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(OlfaBindDataset(augment_pairs(train_p)), batch_size=8, shuffle=True)
    val_loader = DataLoader(OlfaBindDataset(val_p), batch_size=8)
    model = model_cls().to(device)
    
    if hasattr(model, 'input_layer'):
        # Physics needs very gentle lr; sim_head needs wider init
        physics_params = list(model.input_layer.parameters()) + list(model.physics.parameters())
        head_params = [p for p in model.parameters() if not any(p is pp for pp in physics_params)]
        opt = optim.Adam([
            {'params': physics_params, 'lr': 1e-5},
            {'params': head_params, 'lr': lr},
        ], weight_decay=1e-4)
        # Wider sim_head init: initial output should cover [0, 1] range
        if hasattr(model, 'sim_head'):
            for m in model.sim_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=2.0)
        clip_val = 0.5
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        clip_val = 1.0
    
    # Early stopping
    best_r, best_state, patience, no_improve = 0.0, None, 10, 0
    
    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            y = model(b['fp_a'].to(device), b['mask_a'].to(device),
                      b['fp_b'].to(device), b['mask_b'].to(device))
            tgt = b['sim'].to(device)
            loss = F.mse_loss(y, tgt)
            if hasattr(model, 'input_layer'):
                loss = loss + 0.01 * model.input_layer.get_sparsity_loss()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            opt.step()
        
        # Validation & early stopping
        r_val = eval_model(model, val_loader)
        if r_val > best_r:
            best_r = r_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_r

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
# EXP 1: 5-seed x 5-fold CV
# ======================================================================
print("\n" + "="*60)
print("EXP 1: OlfaBind vs Baselines (5-seed x 5-fold CV)")
print("="*60)

results = {}
SEEDS = [42, 123, 456, 789, 2024]

# Non-parametric baselines (no training needed)
max_r = max_pool_baseline(snitz_all)
mean_r = mean_pool_baseline(snitz_all)
results['max_pool'] = {'mean_r': max_r, 'std_r': 0.0, 'n_params': 0}
results['mean_pool'] = {'mean_r': mean_r, 'std_r': 0.0, 'n_params': 0}
print(f"  Max Pool:  r={max_r:.4f}")
print(f"  Mean Pool: r={mean_r:.4f}")

# Trained models
for name, model_cls in [
    ('olfabind', OlfaBindModel),
    ('std_attention', StdAttentionModel),
]:
    print(f"\n--- {name} ---")
    all_rs = []; t0 = time.time()
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (ti, vi) in enumerate(kf.split(range(len(snitz_all)))):
            tp = [snitz_all[i] for i in ti]; vp = [snitz_all[i] for i in vi]
            _, r = train_and_eval(model_cls, tp, vp, epochs=50, lr=5e-4, seed=seed*100+fold)
            all_rs.append(r)
            print(f"  Seed={seed} Fold={fold+1} r={r:.4f}")

    n_params = sum(p.numel() for p in model_cls().parameters())
    results[name] = {
        'mean_r': float(np.mean(all_rs)),
        'std_r': float(np.std(all_rs)),
        'ci_95': [float(np.percentile(all_rs, 2.5)), float(np.percentile(all_rs, 97.5))],
        'all_r': [float(x) for x in all_rs],
        'n_params': n_params,
        'time_sec': time.time() - t0
    }
    print(f"  => r={np.mean(all_rs):.4f} +/- {np.std(all_rs):.4f}  params={n_params}")

# ======================================================================
# EXP 2: Physics embedding analysis
# ======================================================================
print("\n" + "="*60)
print("EXP 2: Physics Embedding Dimension Analysis")
print("="*60)

# Train on all Snitz, then extract physics embeddings
full_model = OlfaBindModel().to(device)
physics_params = list(full_model.input_layer.parameters()) + list(full_model.physics.parameters())
head_params = [p for p in full_model.parameters() if not any(p is pp for pp in physics_params)]
opt = optim.Adam([
    {'params': physics_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 5e-4},
], weight_decay=1e-4)
# Wider sim_head init
if hasattr(full_model, 'sim_head'):
    for m in full_model.sim_head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2.0)
full_loader = DataLoader(OlfaBindDataset(augment_pairs(snitz_all)), batch_size=8, shuffle=True)
for ep in range(50):
    full_model.train()
    for b in full_loader:
        opt.zero_grad()
        y = full_model(b['fp_a'].to(device), b['mask_a'].to(device),
                       b['fp_b'].to(device), b['mask_b'].to(device))
        tgt = b['sim'].to(device)
        loss = F.mse_loss(y, tgt)
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), 0.5)
        opt.step()

# Extract physics embeddings for each mixture
full_model.eval()
emb_names = ['stability', 'S_E', 'S_R', 'S_C', 'mean_mass', 'mean_vel', 'total_L',
             'displacement', 'init_r', 'final_r', 
             'init_pos_x', 'init_pos_y', 'init_pos_z',
             'final_pos_x', 'final_pos_y', 'final_pos_z',
             'final_vel_x', 'final_vel_y', 'final_vel_z']
dim_correlations = {n: {'pearson': 0.0, 'spearman': 0.0} for n in emb_names}

preds_all, trues_all = [], []
with torch.no_grad():
    for p in snitz_all:
        ds = OlfaBindDataset([p])
        b = ds[0]
        y = full_model(b['fp_a'].unsqueeze(0).to(device), b['mask_a'].unsqueeze(0).to(device),
                       b['fp_b'].unsqueeze(0).to(device), b['mask_b'].unsqueeze(0).to(device))
        preds_all.append(y.item())
        trues_all.append(p['sim'])

# Per-dimension analysis
with torch.no_grad():
    all_embs_a, all_embs_b, all_sims = [], [], []
    for p in snitz_all:
        ds = OlfaBindDataset([p])
        b = ds[0]
        _, ea = full_model.get_physics_embeddings(b['fp_a'].unsqueeze(0).to(device), b['mask_a'].unsqueeze(0).to(device))
        _, eb = full_model.get_physics_embeddings(b['fp_b'].unsqueeze(0).to(device), b['mask_b'].unsqueeze(0).to(device))
        all_embs_a.append(ea.cpu().numpy()[0])
        all_embs_b.append(eb.cpu().numpy()[0])
        all_sims.append(p['sim'])

    embs_a = np.array(all_embs_a)
    embs_b = np.array(all_embs_b)
    sims = np.array(all_sims)

    for i, name in enumerate(emb_names):
        diff = np.abs(embs_a[:, i] - embs_b[:, i])
        if np.std(diff) > 1e-8:
            pr, _ = pearsonr(-diff, sims)
            sr, _ = spearmanr(-diff, sims)
        else:
            pr, sr = 0.0, 0.0
        dim_correlations[name] = {'pearson': float(pr), 'spearman': float(sr)}
        print(f"  {name:15s}: Pearson={pr:.4f}  Spearman={sr:.4f}")

results['dim_analysis'] = dim_correlations

# ======================================================================
# EXP 3: Zero-shot Transfer
# ======================================================================
print("\n" + "="*60)
print("EXP 3: Zero-shot Transfer (Snitz -> Ravia/Bushdid)")
print("="*60)

transfer_results = {}
for tname, tpairs in [('ravia', ravia_pairs), ('bushdid', bushdid_pairs)]:
    if not tpairs: continue
    max_r_t = max_pool_baseline(tpairs)
    tloader = DataLoader(OlfaBindDataset(tpairs), batch_size=16)
    olfabind_r = eval_model(full_model, tloader)
    transfer_results[tname] = {
        'n_pairs': len(tpairs),
        'max_pool_r': float(max_r_t),
        'olfabind_r': float(olfabind_r),
        'delta': float(olfabind_r - max_r_t)
    }
    print(f"  {tname}: MaxPool={max_r_t:.4f}  OlfaBind={olfabind_r:.4f}  delta={olfabind_r-max_r_t:+.4f}")

results['transfer'] = transfer_results

# ======================================================================
# EXP 4: Ablation Study
# ======================================================================
print("\n" + "="*60)
print("EXP 4: Ablation Study")
print("="*60)

ablation_results = {}

# Full model performance (use first seed, 5-fold for speed)
def quick_cv(model_cls, pairs, seed=42, n_folds=5, epochs=50):
    rs = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(kf.split(range(len(pairs)))):
        tp = [pairs[i] for i in ti]; vp = [pairs[i] for i in vi]
        _, r = train_and_eval(model_cls, tp, vp, epochs=epochs, seed=seed*100+fold)
        rs.append(r)
    return float(np.mean(rs)), float(np.std(rs))

# Full OlfaBind
full_r, full_s = quick_cv(OlfaBindModel, snitz_all)
ablation_results['full'] = {'mean_r': full_r, 'std_r': full_s}
print(f"  Full OlfaBind:    r={full_r:.4f} +/- {full_s:.4f}")

# Without gravity (G fixed to 0)
class OlfaBindNoGravity(OlfaBindModel):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.physics.engine.log_G.fill_(-10.0)  # G ~ exp(-10) ~ 0
        self.physics.engine.log_G.requires_grad_(False)

no_g_r, no_g_s = quick_cv(OlfaBindNoGravity, snitz_all)
ablation_results['no_gravity'] = {'mean_r': no_g_r, 'std_r': no_g_s}
print(f"  No Gravity:       r={no_g_r:.4f} +/- {no_g_s:.4f}")

# Without lateral inhibition
class OlfaBindNoInhibition(OlfaBindModel):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.input_layer.slice_array.lateral_inhibition.weight.fill_(0.0)
            self.input_layer.slice_array.lateral_inhibition.weight[0,0,2] = 1.0  # identity
        self.input_layer.slice_array.lateral_inhibition.weight.requires_grad_(False)

no_inhib_r, no_inhib_s = quick_cv(OlfaBindNoInhibition, snitz_all)
ablation_results['no_inhibition'] = {'mean_r': no_inhib_r, 'std_r': no_inhib_s}
print(f"  No Inhibition:    r={no_inhib_r:.4f} +/- {no_inhib_s:.4f}")

# Fewer simulation steps (T=4 vs T=16)
class OlfaBindShortSim(OlfaBindModel):
    def __init__(self):
        super().__init__(n_steps=4)

short_r, short_s = quick_cv(OlfaBindShortSim, snitz_all)
ablation_results['short_sim_T4'] = {'mean_r': short_r, 'std_r': short_s}
print(f"  Short Sim (T=4):  r={short_r:.4f} +/- {short_s:.4f}")

results['ablation'] = ablation_results

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for k in ['max_pool', 'mean_pool', 'std_attention', 'olfabind']:
    v = results[k]
    std = v.get('std_r', 0)
    print(f"  {k:20s}: r={v['mean_r']:.4f} +/- {std:.4f}  params={v.get('n_params',0)}")

print(f"\nAblation:")
for k, v in results['ablation'].items():
    print(f"  {k:20s}: r={v['mean_r']:.4f} +/- {v['std_r']:.4f}")

# Save
out_path = os.path.join(RESULTS_DIR, "v17_olfabind_validation.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"\nResults saved to {out_path}")
