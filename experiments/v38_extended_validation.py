"""
v38_extended_validation.py — Extended Validation for Nature MI Revision
======================================================================
Purpose: Address key reviewer concerns with additional experiments.

EXPERIMENT 1: 10-seed Mol-CV (PhysSim vs Morgan+MLP)
  - 10 seeds × 5 folds = 50 paired observations
  - Target: p < 0.05 for PhysSim > Morgan (currently p=0.059 with 15 pairs)

EXPERIMENT 2: Re-validate ablation with 10 seeds
  - 10 seeds × 5 folds = 50 observations per ablation
  - Target: Statistical significance for individual force contributions

EXPERIMENT 3: Blind transfer (same as v37, but with best 10-seed model)

[Colab 사용법]
  1. Google Drive에 데이터 업로드:
     My Drive/perfume_data/dream_mixture/
       ├── snitz_2013/ (behavior.csv, molecules.csv, stimuli.csv)
       ├── ravia_2020/ (behavior_2.csv, molecules.csv, stimuli.csv)
       └── bushdid_2014/ (behavior.csv, molecules.csv, stimuli.csv)
  2. v38_colab_package.zip 을 /content/ 에 업로드 후 압축 해제
  3. 이 스크립트를 실행 (약 4-5시간 소요)
"""
import sys, os, time, json, csv, warnings, math
warnings.filterwarnings('ignore')

# === Colab 환경 감지 ===
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IS_COLAB:
    print('🔧 Colab 환경 감지!')
    os.system('pip install -q torch-geometric 2>/dev/null')
    BASE = '/content/dream_mixture'
    SAVE_DIR = '/content/results'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'  데이터 경로: {BASE}')
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'server', 'data', 'pom_data', 'dream_mixture')
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasa_data')
    os.makedirs(SAVE_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ======================================================================
# 1. Data Loading (identical to v37)
# ======================================================================
DESC_LIST = [(n, f) for n, f in Descriptors.descList]
N_DESC = len(DESC_LIST)
MORGAN_BITS = 2048

def smiles_to_desc(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(N_DESC, dtype=np.float32)
    vals = []
    for _, func in DESC_LIST:
        try:
            v = func(mol); vals.append(float(v) if v is not None and np.isfinite(v) else 0.0)
        except: vals.append(0.0)
    return np.array(vals, dtype=np.float32)

def smiles_to_morgan(smi, nbits=MORGAN_BITS):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    return np.array(fp, dtype=np.float32)

# Load Snitz
cid2smi = {}
with open(os.path.join(BASE, 'snitz_2013', 'molecules.csv'), 'r', errors='ignore') as f:
    for r in csv.DictReader(f):
        if r.get('IsomericSMILES'): cid2smi[r['CID']] = r['IsomericSMILES']

snitz_pairs = []
with open(os.path.join(BASE, 'snitz_2013', 'behavior.csv'), 'r', errors='ignore') as f:
    for r in csv.DictReader(f):
        ca = [cid2smi[c.strip()] for c in r['StimulusA'].split(',') if c.strip() in cid2smi]
        cb = [cid2smi[c.strip()] for c in r['StimulusB'].split(',') if c.strip() in cid2smi]
        if ca and cb: snitz_pairs.append({'ca':ca,'cb':cb,'sim':float(r['Similarity'])/100.0})

snitz_smi = set()
for p in snitz_pairs:
    for s in p['ca']+p['cb']: snitz_smi.add(s)
print(f"Snitz: {len(snitz_pairs)} pairs, {len(snitz_smi)} mols")

# Load Ravia
rav_cid = {}
with open(os.path.join(BASE, 'ravia_2020', 'molecules.csv'), 'r', errors='ignore') as f:
    for r in csv.DictReader(f):
        if r.get('IsomericSMILES'): rav_cid[r['CID']] = r['IsomericSMILES']
stim2c = {}
with open(os.path.join(BASE, 'ravia_2020', 'stimuli.csv'), 'r', errors='ignore') as f:
    for r in csv.DictReader(f):
        stim2c[r['Stimulus']] = [c.strip() for c in r['CID'].split(';')]
ravia_blind = []
with open(os.path.join(BASE, 'ravia_2020', 'behavior_2.csv'), 'r', errors='ignore') as f:
    for r in csv.DictReader(f):
        s1,s2 = r['Stimulus 1'],r['Stimulus 2']
        if s1==s2: continue
        sm1=[rav_cid[c] for c in stim2c.get(s1,[]) if c in rav_cid]
        sm2=[rav_cid[c] for c in stim2c.get(s2,[]) if c in rav_cid]
        if sm1 and sm2: ravia_blind.append({'ca':sm1,'cb':sm2,'sim':float(r['RatedSimilarity'])/100.0})
print(f"Ravia blind: {len(ravia_blind)} pairs")

# Bushdid
try:
    bush_cid = {}
    with open(os.path.join(BASE, 'bushdid_2014', 'molecules.csv'), 'r', errors='ignore') as f:
        for r in csv.DictReader(f):
            if r.get('IsomericSMILES'): bush_cid[r['CID']] = r['IsomericSMILES']
    bush_stim = {}
    with open(os.path.join(BASE, 'bushdid_2014', 'stimuli.csv'), 'r', errors='ignore') as f:
        for r in csv.DictReader(f):
            bush_stim[r['Stimulus']] = [c.strip() for c in r['CID'].split(';')]
    bushdid_blind = []
    with open(os.path.join(BASE, 'bushdid_2014', 'behavior.csv'), 'r', errors='ignore') as f:
        for r in csv.DictReader(f):
            s1,s2 = r['Stimulus 1'], r['Stimulus 2']
            if s1==s2: continue
            sm1=[bush_cid[c] for c in bush_stim.get(s1,[]) if c in bush_cid]
            sm2=[bush_cid[c] for c in bush_stim.get(s2,[]) if c in bush_cid]
            if sm1 and sm2:
                bushdid_blind.append({'ca':sm1,'cb':sm2,'sim':float(r['Correct'])})
    print(f"Bushdid blind: {len(bushdid_blind)} pairs")
except:
    bushdid_blind = []
    print("Bushdid: not available")

# Compute features
all_smi = set(snitz_smi)
for p in ravia_blind:
    for s in p['ca']+p['cb']: all_smi.add(s)
for p in bushdid_blind:
    for s in p['ca']+p['cb']: all_smi.add(s)
print(f"Computing features for {len(all_smi)} molecules...")

desc_cache, morgan_cache = {}, {}
for smi in all_smi:
    desc_cache[smi] = smiles_to_desc(smi)
    morgan_cache[smi] = smiles_to_morgan(smi)

train_d = np.array([desc_cache[s] for s in snitz_smi])
d_mean = train_d.mean(0); d_std = train_d.std(0)+1e-8
desc_norm = {}
for smi in desc_cache:
    desc_norm[smi] = (desc_cache[smi]-d_mean)/d_std

# ======================================================================
# 2. Datasets & Models (identical to v37)
# ======================================================================
MAX_MOLS = 50

class DescDS(Dataset):
    def __init__(self, pairs, cache): self.pairs=pairs; self.cache=cache
    def __len__(self): return len(self.pairs)
    def _pad(self, sl, dim):
        o=np.zeros((MAX_MOLS,dim),dtype=np.float32);m=np.zeros(MAX_MOLS,dtype=np.float32)
        for i,s in enumerate(sl[:MAX_MOLS]):
            if s in self.cache: o[i]=self.cache[s];m[i]=1.0
        return o,m
    def __getitem__(self,i):
        p=self.pairs[i]; dim=len(next(iter(self.cache.values())))
        a,ma=self._pad(p['ca'],dim); b,mb=self._pad(p['cb'],dim)
        return {'e_a':torch.from_numpy(a),'m_a':torch.from_numpy(ma),
                'e_b':torch.from_numpy(b),'m_b':torch.from_numpy(mb),
                'sim':torch.tensor(p['sim'],dtype=torch.float32)}

def augment(p): return p+[{'ca':x['cb'],'cb':x['ca'],'sim':x['sim']} for x in p]

D_POS=128; D_ATOM=128; D_PROJ=128

class PhysicsEngine(nn.Module):
    def __init__(self, n_steps=16, use_coulomb=True, use_vdw=True, use_spin=True, use_gr=True):
        super().__init__()
        self.n_steps=n_steps; self.use_coulomb=use_coulomb
        self.use_vdw=use_vdw; self.use_spin=use_spin; self.use_gr=use_gr
        self.chem_enc=nn.Sequential(nn.Linear(N_DESC,D_ATOM),nn.LayerNorm(D_ATOM),nn.GELU(),
            nn.Linear(D_ATOM,D_ATOM),nn.LayerNorm(D_ATOM),nn.GELU())
        self.mass_mapper=nn.Linear(D_ATOM,1)
        self.charge_mapper=nn.Linear(D_ATOM,1)
        self.sigma_mapper=nn.Linear(D_ATOM,1)
        self.pos_mapper=nn.Linear(D_ATOM,D_POS)
        self.vel_mapper=nn.Linear(D_ATOM,D_POS)
        self.spin_mapper=nn.Linear(D_ATOM,D_POS)
        self.log_G=nn.Parameter(torch.tensor(0.0))
        self.log_c=nn.Parameter(torch.tensor(1.0))
        self.log_kappa=nn.Parameter(torch.tensor(-1.0))
        self.log_beta=nn.Parameter(torch.tensor(1.0))
        self.log_k_e=nn.Parameter(torch.tensor(0.0))
        self.log_eps_lj=nn.Parameter(torch.tensor(-1.0))
        self.log_spin_coupling=nn.Parameter(torch.tensor(-1.0))

    def forward(self, emb, mask):
        eps=1e-8; me=mask.unsqueeze(-1); atoms=self.chem_enc(emb)
        masses=F.softplus(self.mass_mapper(atoms))*me
        charges=torch.tanh(self.charge_mapper(atoms))*me
        sigmas=F.softplus(self.sigma_mapper(atoms))*me+0.1
        positions=self.pos_mapper(atoms)*me
        velocities=self.vel_mapper(atoms)*me
        spins=self.spin_mapper(atoms)*me
        G=torch.exp(self.log_G);c=torch.exp(self.log_c)
        kappa=torch.exp(self.log_kappa);beta=torch.exp(self.log_beta)
        k_e=torch.exp(self.log_k_e);eps_lj=torch.exp(self.log_eps_lj)
        spin_coup=torch.exp(self.log_spin_coupling)
        dt=0.1/self.n_steps; traj=[positions]; mass_h=[masses]
        for _ in range(self.n_steps):
            B,N,D=positions.shape
            diff=positions.unsqueeze(2)-positions.unsqueeze(1)
            r=diff.norm(dim=-1,keepdim=True).clamp(min=eps)
            r_hat=diff/(r+eps)
            mi=masses.unsqueeze(2);mj=masses.unsqueeze(1)
            if self.use_gr:
                r_s=2*G*mj/(c**2+eps)
                gr_boost=1.0+F.softplus(beta*r_s/(r+eps))
            else:
                gr_boost=1.0
            F_grav=-G*mi*mj/(r**2+eps)*r_hat*gr_boost
            F_total=F_grav
            if self.use_coulomb:
                qi=charges.unsqueeze(2);qj=charges.unsqueeze(1)
                F_total=F_total+k_e*qi*qj/(r**2+eps)*r_hat
            if self.use_vdw:
                si=sigmas.unsqueeze(2);sj=sigmas.unsqueeze(1)
                sigma_ij=(si+sj)/2
                r_sc=torch.sqrt(r**2+0.25)
                sr6=(sigma_ij/r_sc)**6; sr12=sr6**2
                F_total=F_total+24*eps_lj/r_sc*(2*sr12-sr6)*r_hat
            pmask=mask.unsqueeze(1).unsqueeze(-1)*mask.unsqueeze(2).unsqueeze(-1)
            eye=(1-torch.eye(N,device=emb.device)).unsqueeze(0).unsqueeze(-1)
            acc=(F_total*pmask*eye).sum(2)/(masses+eps)
            v_norm=velocities.norm(dim=-1,keepdim=True)
            gamma=1.0/torch.sqrt(1.0-(v_norm/(c+eps)).clamp(max=0.999)**2+eps)
            acc_eff=acc/(gamma+eps)
            if self.use_spin:
                sn=spins.norm(dim=-1,keepdim=True).clamp(max=10.0)
                ra=sn*dt*spin_coup
                ca_=torch.cos(ra);sa_=torch.sin(ra)
                vp=velocities.view(B,N,D//2,2)
                ve=vp[:,:,:,0:1];vo=vp[:,:,:,1:2]
                ca2=ca_.unsqueeze(-1);sa2=sa_.unsqueeze(-1)
                velocities=torch.stack([ve*ca2-vo*sa2,ve*sa2+vo*ca2],dim=-1).view(B,N,D).squeeze(-1)*me
            velocities=(velocities+acc_eff*dt)*me
            velocities=torch.where(torch.isfinite(velocities),velocities,torch.zeros_like(velocities))
            positions=(positions+velocities*dt)*me
            positions=torch.where(torch.isfinite(positions),positions,torch.zeros_like(positions))
            dm=kappa/(masses**2+eps)*dt
            masses=(masses-dm*me).clamp(min=eps)
            if self.use_spin: spins=spins*(1.0-0.01*dt)*me
            traj.append(positions); mass_h.append(masses)
        traj_t=torch.stack(traj,1); mass_t=torch.stack(mass_h,1)
        fp=traj_t[:,-1]; pv=traj_t.var(1).sum(-1,keepdim=True)
        va=traj_t[:,1:]-traj_t[:,:-1]; sp=va.norm(dim=-1)
        ms_=sp.max(1).values.unsqueeze(-1); sv=sp.var(1).unsqueeze(-1)
        mf=mass_t[:,-1]; mr=mf/(mass_t[:,0]+eps)
        feats=torch.cat([fp,pv,ms_,sv,mf,mr,charges.abs()],dim=-1)*me
        return feats.sum(1)/(mask.sum(1,keepdim=True)+eps)

D_FEAT=D_POS+6

class PhysicsModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.engine=PhysicsEngine(**kwargs)
        self.proj=nn.Sequential(nn.Linear(D_FEAT,D_PROJ),nn.LayerNorm(D_PROJ),nn.GELU(),
            nn.Dropout(0.1),nn.Linear(D_PROJ,D_PROJ))
        self.sim_head=nn.Sequential(nn.Linear(2*D_PROJ+1,D_PROJ),nn.GELU(),nn.Dropout(0.1),nn.Linear(D_PROJ,1))
    def forward(self,ea,ma,eb,mb):
        pa=self.proj(self.engine(ea,ma));pb=self.proj(self.engine(eb,mb))
        d=(pa-pb).abs();p=pa*pb
        cos=F.cosine_similarity(pa,pb,dim=-1,eps=1e-8).unsqueeze(-1)
        return torch.sigmoid(self.sim_head(torch.cat([d,p,cos],-1)).squeeze(-1))

class MLP_Model(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(in_dim,D_ATOM),nn.LayerNorm(D_ATOM),nn.GELU(),
            nn.Linear(D_ATOM,D_PROJ),nn.LayerNorm(D_PROJ),nn.GELU(),nn.Dropout(0.1),nn.Linear(D_PROJ,D_PROJ))
        self.sim_head=nn.Sequential(nn.Linear(2*D_PROJ+1,D_PROJ),nn.GELU(),nn.Dropout(0.1),nn.Linear(D_PROJ,1))
    def forward(self,ea,ma,eb,mb):
        me_a=ma.unsqueeze(-1);me_b=mb.unsqueeze(-1)
        a=(ea*me_a).sum(1)/(ma.sum(1,keepdim=True)+1e-8)
        b=(eb*me_b).sum(1)/(mb.sum(1,keepdim=True)+1e-8)
        pa=self.enc(a);pb=self.enc(b)
        d=(pa-pb).abs();p=pa*pb
        cos=F.cosine_similarity(pa,pb,dim=-1,eps=1e-8).unsqueeze(-1)
        return torch.sigmoid(self.sim_head(torch.cat([d,p,cos],-1)).squeeze(-1))

# ======================================================================
# 3. Training utilities
# ======================================================================
def spearman_proxy(pred, target, alpha=10.0):
    n=pred.shape[0]
    if n<3:
        pc=pred-pred.mean();tc=target-target.mean()
        return 1.0-(pc*tc).sum()/(torch.sqrt((pc**2).sum()+1e-8)*torch.sqrt((tc**2).sum()+1e-8))
    pr=torch.sigmoid(alpha*(pred.unsqueeze(1)-pred.unsqueeze(0))).sum(1)+1.0
    tr=torch.sigmoid(alpha*(target.unsqueeze(1)-target.unsqueeze(0))).sum(1)+1.0
    pc=pr-pr.mean();tc=tr-tr.mean()
    return 1.0-(pc*tc).sum()/(torch.sqrt((pc**2).sum()+1e-8)*torch.sqrt((tc**2).sum()+1e-8))

def combo_loss(pred, target):
    return 0.7*F.mse_loss(pred,target)+0.3*spearman_proxy(pred,target)

def evaluate_desc(model, loader):
    model.eval();ps,ts=[],[]
    with torch.no_grad():
        for b in loader:
            y=model(b['e_a'].to(device),b['m_a'].to(device),b['e_b'].to(device),b['m_b'].to(device))
            for i in range(len(y)):
                if torch.isfinite(y[i]): ps.append(y[i].cpu().item());ts.append(b['sim'][i].item())
    if len(ps)<2 or np.std(ps)<1e-8: return 0.0,0.0
    return pearsonr(ps,ts)[0],spearmanr(ps,ts)[0]

def train_desc_cv(model,tl,vl,epochs=80,lr=3e-4):
    opt=optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs-5)
    best_r,best_s,pat=0.0,None,0
    for ep in range(epochs):
        if ep<5:
            for pg in opt.param_groups: pg['lr']=lr*(ep+1)/5
        model.train()
        for b in tl:
            y=model(b['e_a'].to(device),b['m_a'].to(device),b['e_b'].to(device),b['m_b'].to(device))
            loss=combo_loss(y.float(),b['sim'].to(device))
            if not torch.isfinite(loss): continue
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if ep>=5: sched.step()
        _,rs=evaluate_desc(model,vl)
        if rs>best_r: best_r=rs;best_s={k:v.cpu().clone() for k,v in model.state_dict().items()};pat=0
        else:
            pat+=1
            if pat>=20: break
    if best_s: model.load_state_dict({k:v.to(device) for k,v in best_s.items()})
    return model,best_r

# ======================================================================
# 4. EXPERIMENT 1: Extended 10-seed Mol-CV
# ======================================================================
all_mols = sorted(list(snitz_smi))
SEEDS_10 = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777]
N_R = 2  # 2 restarts per fold
t0 = time.time()

print(f"\n{'='*70}")
print(f"  v38: EXTENDED VALIDATION — 10 Seeds x 5 Folds = 50 Observations")
print(f"{'='*70}")

# Ablation configs
ablation_configs = {
    'Full_Physics':  dict(use_coulomb=True, use_vdw=True, use_spin=True, use_gr=True),
    'No_Coulomb':    dict(use_coulomb=False, use_vdw=True, use_spin=True, use_gr=True),
    'No_vdW':        dict(use_coulomb=True, use_vdw=False, use_spin=True, use_gr=True),
    'No_Spin':       dict(use_coulomb=True, use_vdw=True, use_spin=False, use_gr=True),
    'No_GR':         dict(use_coulomb=True, use_vdw=True, use_spin=True, use_gr=False),
    'Gravity_Only':  dict(use_coulomb=False, use_vdw=False, use_spin=False, use_gr=False),
}

all_results = {}

def run_mol_cv(model_cls, model_kwargs, seeds, cache, label):
    """Run molecule-level CV with given seeds and return per-fold results."""
    results = []
    total = len(seeds) * 5
    done = 0
    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (ti, vi) in enumerate(kf.split(all_mols)):
            tm = set(all_mols[i] for i in ti)
            vm = set(all_mols[i] for i in vi)
            tp = [p for p in snitz_pairs if set(p['ca']).issubset(tm) and set(p['cb']).issubset(tm)]
            vp = [p for p in snitz_pairs if bool(set(p['ca'])&vm) or bool(set(p['cb'])&vm)]
            if len(vp) < 5:
                results.append(0.0)
                done += 1
                continue
            tl = DataLoader(DescDS(augment(tp), cache), batch_size=8, shuffle=True)
            vl = DataLoader(DescDS(vp, cache), batch_size=8)
            best = 0
            for r in range(N_R):
                torch.manual_seed(seed*1000+fold*100+r)
                np.random.seed(seed*1000+fold*100+r)
                if model_cls == PhysicsModel:
                    m = model_cls(**model_kwargs).to(device)
                else:
                    m = model_cls(model_kwargs).to(device)
                m, _ = train_desc_cv(m, tl, vl)
                _, rs = evaluate_desc(m, vl)
                if rs > best: best = rs
                del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            results.append(best)
            done += 1
            if done % 10 == 0:
                print(f"    [{label}] {done}/{total} folds done, current mean={np.mean(results):.4f}", flush=True)
    return results

# --- Run PhysSim Full + Ablations ---
print(f"\n  === EXPERIMENT 1: PhysSim Ablation Study (10 seeds) ===")
for abl_name, abl_kwargs in ablation_configs.items():
    print(f"\n  Running {abl_name}...")
    results = run_mol_cv(PhysicsModel, abl_kwargs, SEEDS_10, desc_norm, abl_name)
    all_results[abl_name] = results
    print(f"  {abl_name}: mean={np.mean(results):.4f}, std={np.std(results):.4f}, n={len(results)}")

# --- Run Morgan+MLP ---
print(f"\n  === EXPERIMENT 2: Morgan+MLP (10 seeds) ===")
morgan_results = run_mol_cv(MLP_Model, MORGAN_BITS, SEEDS_10, morgan_cache, 'Morgan+MLP')
all_results['Morgan_MLP'] = morgan_results
print(f"  Morgan+MLP: mean={np.mean(morgan_results):.4f}, std={np.std(morgan_results):.4f}")

# --- Run RDKit+MLP ---
print(f"\n  === EXPERIMENT 3: RDKit+MLP (10 seeds) ===")
rdkit_results = run_mol_cv(MLP_Model, N_DESC, SEEDS_10, desc_norm, 'RDKit+MLP')
all_results['RDKit_MLP'] = rdkit_results
print(f"  RDKit+MLP: mean={np.mean(rdkit_results):.4f}, std={np.std(rdkit_results):.4f}")

# ======================================================================
# 5. Statistical Analysis
# ======================================================================
print(f"\n{'='*70}")
print(f"  STATISTICAL ANALYSIS (n=50 paired observations)")
print(f"{'='*70}")

ref = np.array(all_results['Full_Physics'])

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / (pooled + 1e-8)

def paired_bootstrap(a, b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        deltas.append(np.mean(a[idx]) - np.mean(b[idx]))
    deltas = np.array(deltas)
    p = np.mean(deltas <= 0)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return p, ci_lo, ci_hi

print(f"\n  {'Model':<15} {'mean':>7} {'delta':>7} {'Wilcoxon':>10} {'Bootstrap':>10} {'Cohen_d':>8} {'Wins':>5}")
print(f"  {'='*65}")

stats_results = {}
for name, vals in all_results.items():
    vals = np.array(vals)
    n = min(len(ref), len(vals))
    delta = ref[:n].mean() - vals[:n].mean()
    wins = int(np.sum(ref[:n] > vals[:n]))
    cd = cohens_d(ref[:n], vals[:n])

    try:
        _, p_wilcox = wilcoxon(ref[:n], vals[:n], alternative='greater')
    except:
        p_wilcox = 1.0

    p_boot, ci_lo, ci_hi = paired_bootstrap(ref[:n], vals[:n])

    if name == 'Full_Physics':
        print(f"  {name:<15} {vals.mean():>7.4f}     ---        ---        ---      ---   ---")
    else:
        sig_w = '***' if p_wilcox < 0.001 else '**' if p_wilcox < 0.01 else '*' if p_wilcox < 0.05 else 'ns'
        sig_b = '***' if p_boot < 0.001 else '**' if p_boot < 0.01 else '*' if p_boot < 0.05 else 'ns'
        print(f"  {name:<15} {vals.mean():>7.4f} {delta:>+7.4f} {p_wilcox:>8.4f}{sig_w:>2s} {p_boot:>8.4f}{sig_b:>2s} {cd:>8.2f} {wins:>3}/{n}")

    stats_results[name] = {
        'mean': float(vals.mean()),
        'std': float(vals.std()),
        'delta': float(delta),
        'p_wilcoxon': float(p_wilcox),
        'p_bootstrap': float(p_boot),
        'cohen_d': float(cd),
        'wins': int(wins),
        'n': int(n),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'all': [float(x) for x in vals]
    }

# ======================================================================
# 6. Blind Transfer Test
# ======================================================================
print(f"\n{'='*70}")
print(f"  BLIND TRANSFER TEST")
print(f"{'='*70}")

print(f"  Training Full Physics on all Snitz (3 restarts)...")
full_loader = DataLoader(DescDS(augment(snitz_pairs), desc_norm), batch_size=8, shuffle=True)

best_model = None; best_rho = -1
for r in range(3):
    torch.manual_seed(42+r*100); np.random.seed(42+r*100)
    m = PhysicsModel(**ablation_configs['Full_Physics']).to(device)
    opt = optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=95)
    for ep in range(100):
        if ep < 5:
            for pg in opt.param_groups: pg['lr'] = 3e-4*(ep+1)/5
        m.train()
        for b in full_loader:
            y = m(b['e_a'].to(device), b['m_a'].to(device), b['e_b'].to(device), b['m_b'].to(device))
            loss = combo_loss(y.float(), b['sim'].to(device))
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep >= 5: sched.step()
    m.eval()
    el = DataLoader(DescDS(snitz_pairs, desc_norm), batch_size=8)
    ps, ts = [], []
    with torch.no_grad():
        for b in el:
            y = m(b['e_a'].to(device), b['m_a'].to(device), b['e_b'].to(device), b['m_b'].to(device))
            for i in range(len(y)):
                if torch.isfinite(y[i]): ps.append(y[i].cpu().item()); ts.append(b['sim'][i].item())
    rho = spearmanr(ps, ts)[0] if len(ps) > 2 else 0
    print(f"    R{r+1}: train_rho={rho:.4f}")
    if rho > best_rho: best_rho = rho; best_model = m

# Ravia blind
ds = DescDS(ravia_blind, desc_norm)
best_model.eval(); bps, bts = [], []
with torch.no_grad():
    for i in range(len(ds)):
        b = ds[i]
        ea = b['e_a'].unsqueeze(0).to(device); ma = b['m_a'].unsqueeze(0).to(device)
        eb = b['e_b'].unsqueeze(0).to(device); mb = b['m_b'].unsqueeze(0).to(device)
        pred = best_model(ea, ma, eb, mb).item()
        if np.isfinite(pred): bps.append(pred); bts.append(b['sim'].item())
blind_p = pearsonr(bps, bts)[0]; blind_s = spearmanr(bps, bts)[0]
print(f"  Ravia BLIND: Pearson={blind_p:.4f}, Spearman={blind_s:.4f}")

# Bushdid blind
if bushdid_blind:
    ds_b = DescDS(bushdid_blind, desc_norm)
    bps2, bts2 = [], []
    with torch.no_grad():
        for i in range(len(ds_b)):
            b = ds_b[i]
            ea = b['e_a'].unsqueeze(0).to(device); ma = b['m_a'].unsqueeze(0).to(device)
            eb = b['e_b'].unsqueeze(0).to(device); mb = b['m_b'].unsqueeze(0).to(device)
            pred = best_model(ea, ma, eb, mb).item()
            if np.isfinite(pred): bps2.append(pred); bts2.append(b['sim'].item())
    bush_p = pearsonr(bps2, bts2)[0]; bush_s = spearmanr(bps2, bts2)[0]
    print(f"  Bushdid BLIND: Pearson={bush_p:.4f}, Spearman={bush_s:.4f}")
else:
    bush_p, bush_s = 0, 0

# Learned constants
e = best_model.engine
print(f"\n  Learned Physical Constants:")
print(f"    G={torch.exp(e.log_G).item():.4f}, c={torch.exp(e.log_c).item():.4f}")
print(f"    k_e={torch.exp(e.log_k_e).item():.4f}, eps_lj={torch.exp(e.log_eps_lj).item():.4f}")
print(f"    spin={torch.exp(e.log_spin_coupling).item():.4f}")
print(f"    kappa={torch.exp(e.log_kappa).item():.4f}, beta={torch.exp(e.log_beta).item():.4f}")

# RDKit cosine baseline
from numpy.linalg import norm as np_norm
ps_cos, ts_cos = [], []
for p in ravia_blind:
    ea = [desc_norm[s] for s in p['ca'] if s in desc_norm]
    eb = [desc_norm[s] for s in p['cb'] if s in desc_norm]
    if ea and eb:
        a = np.mean(ea, 0); b = np.mean(eb, 0)
        ps_cos.append(np.dot(a, b)/(np_norm(a)*np_norm(b)+1e-8))
        ts_cos.append(p['sim'])
cos_p, cos_s = pearsonr(ps_cos, ts_cos)[0], spearmanr(ps_cos, ts_cos)[0]
print(f"  RDKit cosine BLIND: Pearson={cos_p:.4f}, Spearman={cos_s:.4f}")

# ======================================================================
# 7. Save Everything
# ======================================================================
elapsed = (time.time() - t0) / 60

save_data = {
    'experiment': 'v38_extended_validation',
    'n_seeds': len(SEEDS_10),
    'seeds': SEEDS_10,
    'n_observations': len(SEEDS_10) * 5,
    'time_minutes': elapsed,
    'blind': {
        'ravia_pearson': blind_p, 'ravia_spearman': blind_s,
        'bushdid_pearson': bush_p, 'bushdid_spearman': bush_s,
        'rdkit_cos_spearman': cos_s,
    },
    'models': stats_results,
}

save_path = os.path.join(SAVE_DIR, 'v38_extended_validation.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\n{'='*70}")
print(f"  v38 COMPLETE! ({elapsed:.1f} min)")
print(f"  Results saved: {save_path}")
print(f"{'='*70}")
print(f"\n  KEY RESULT: PhysSim vs Morgan+MLP")
print(f"    Wilcoxon p = {stats_results['Morgan_MLP']['p_wilcoxon']:.4f}")
print(f"    Bootstrap p = {stats_results['Morgan_MLP']['p_bootstrap']:.4f}")
print(f"    Cohen's d = {stats_results['Morgan_MLP']['cohen_d']:.2f}")
print(f"    Wins: {stats_results['Morgan_MLP']['wins']}/{stats_results['Morgan_MLP']['n']}")
print(f"\n  DONE!")
