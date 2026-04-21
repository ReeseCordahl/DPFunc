# Running DPFunc

Running DPFunc required significantly more troubleshooting and setup compared to previous tools. Below is a summary of my workflow and key challenges when generating `.pkl` files.

## Key Notes

1. **Environment Setup**
   - Created and activated a dedicated Conda environment for DPFunc

2. **Directory Organization**
   - Created structured directories for:
     - Inputs (`.fasta`, `.pdb`)
     - Embeddings
     - Graphs
     - Results
     - Scripts

3. **Input Preparation**
   - Copied `.pdb` and `.fasta` files for 8 proteins generated from ASAM2 and BioEmu into organized input folders

4. **Dependencies**
   - Installed multiple required packages (PyTorch, DGL, Biopython, etc.) to get the pipeline running

5. **Script Debugging**
   - Modified and patched the `old_process_structure` script both locally and within DPFunc to correctly process protein data

6. **Notebook Issues**
   - Encountered issues running the provided Jupyter Notebook and instead relied on custom scripts

---

## Example Workflow

### Navigate and Request GPU Node
```bash
cd ~/DPFunc

salloc --gres=gpu:1 --mem=64G --time=01:00:00
```

### Set Up Conda Environment
```bash
module purge
module load Miniforge3

conda create --name dpfunc
conda activate dpfunc
```

---

## Directory Setup

```bash
mkdir -p ~/dpfunc_data/{fasta,structures}

ln -s ~/test-my-protein/inputs/*.fasta ~/dpfunc_data/fasta/
ln -s ~/asam2_proteins/*.pdb ~/dpfunc_data/structures/
```

---

## Install Required Packages

```bash
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install --upgrade packaging setuptools
pip install torchdata==0.7.1

pip install torch==2.1.2+cu118 torchvision \
  --index-url https://download.pytorch.org/whl/cu118

pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install torchdata pandas pyyaml packaging setuptools
pip install biopython joblib ruamel.yaml
```

---

## Copy Input Files

```bash
cp ~/test-my-protein/inputs/3h3b.fasta data_dpfunc/fasta/
cp ~/test-my-protein/inputs/3h3b.pdb data_dpfunc/pdb/
```

---

## Generate Graph Files

```bash
python scripts/pdb_to_graph.py
```

---

## Generate ESM Embeddings

### Script: `generate_esm_embeddings.py`

```python
import os
import pickle
import torch
from esm import pretrained
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

FASTA_DIR = "inputs/fasta"
OUT_DIR = "esm"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading ESM-2 model...")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

for fasta_file in sorted(os.listdir(FASTA_DIR)):
    if not fasta_file.endswith(".fasta"):
        continue

    protein_name = Path(fasta_file).stem
    print(f"Processing {protein_name}...")

    fasta_path = os.path.join(FASTA_DIR, fasta_file)
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append((record.id, str(record.seq)))

    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

    embeddings = []
    for i, (_, seq) in enumerate(sequences):
        seq_len = len(seq)
        emb = token_representations[i, 1:seq_len+1].cpu()
        embeddings.append(emb)

    out_path = os.path.join(OUT_DIR, protein_name + ".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings → {out_path}\n")
```

### Run Script
```bash
python scripts/generate_esm_embeddings.py
```

---

## Patch Processing Script (Local)

### `patched_old_process_structure.py`

```python
import os
import pickle
import torch 

def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
   
    if not hasattr(g, 'edata'):
        g.edata = {}
    if not hasattr(g, 'ndata'):
        g.ndata = {}
    return g

def load_esm(esm_path):
    with open(esm_path, "rb") as f:
        embeddings = pickle.load(f)
    if embeddings is None or len(embeddings) == 0:
        embeddings = []
    return embeddings

def process_protein(protein_name, graph_dir="graphs", esm_dir="esm"):
    graph_path = os.path.join(graph_dir, protein_name + ".pkl")
    esm_path = os.path.join(esm_dir, protein_name + ".pkl")

    g = load_graph(graph_path)
    esm_emb = load_esm(esm_path)

    if "residue_feature" not in g.ndata:
        if esm_emb and len(esm_emb[0]) == g.num_nodes():
            g.ndata["residue_feature"] = esm_emb[0]
        else:
            feature_dim = esm_emb[0].shape[1] if esm_emb else 128
            g.ndata["residue_feature"] = torch.zeros((g.num_nodes(), feature_dim))

    return g, esm_emb
```

---

## Patch Processing Script (DPFunc)

### `old_process_structure_Reese.py`

```python

import os
import pickle
from tqdm.auto import tqdm
import torch
import dgl

GRAPH_DIR = "/mnt/home/cordahlr/dpfunc_run/graphs"      # Graph pickles
ESM_DIR   = "/mnt/home/cordahlr/dpfunc_run/esm"         # ESM embedding pickles
OUT_DIR   = "/mnt/home/cordahlr/dpfunc_run/results"     # Output predictions
PROTEIN_LIST_FILE = "/mnt/home/cordahlr/dpfunc_run/protein_list.txt"  # List of proteins

os.makedirs(OUT_DIR, exist_ok=True)

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_pkl(file_path, val):
    with open(file_path, "wb") as f:
        pickle.dump(val, f)

def ensure_residue_features(graph, esm_emb):
    """
    Assign per-residue ESM embeddings to graph.ndata['residue_feature'].
    If missing or sizes mismatch, create zeros.
    """
    num_nodes = graph.num_nodes()

    if "residue_feature" not in graph.ndata:
        if esm_emb and len(esm_emb[0]) == num_nodes:
            graph.ndata["residue_feature"] = esm_emb[0]
        else:
            feature_dim = esm_emb[0].shape[1] if esm_emb else 1280
            graph.ndata["residue_feature"] = torch.zeros((num_nodes, feature_dim))

    return graph

def process_protein(protein_name):
    graph_path = os.path.join(GRAPH_DIR, f"{protein_name}.pkl")
    esm_path   = os.path.join(ESM_DIR, f"{protein_name}.pkl")

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph pickle not found: {graph_path}")
    if not os.path.exists(esm_path):
        raise FileNotFoundError(f"ESM embedding pickle not found: {esm_path}")

    g = read_pkl(graph_path)
    esm_emb = read_pkl(esm_path)

    g = ensure_residue_features(g, esm_emb)

    out_path = os.path.join(OUT_DIR, f"{protein_name}_processed.pkl")
    save_pkl(out_path, g)

    print(f"{protein_name}: nodes={g.num_nodes()}, residue_feature shape={g.ndata['residue_feature'].shape}")
    print(f"Saved processed graph → {out_path}")

    return g

if __name__ == "__main__":

    with open(PROTEIN_LIST_FILE, "r") as f:
        protein_list = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Processing {len(protein_list)} proteins...")

    for protein in tqdm(protein_list):
        process_protein(protein)
```

---

## Run Final Processing

```bash
python ~/DPFunc/DataProcess/old_process_structure_Reese.py
```

---

## Summary

- DPFunc required extensive setup, debugging, and dependency management  
- Custom scripts were necessary to properly generate `.pkl` graph and embedding files  
- GPU access and correct environment configuration were critical for successful execution  
- Despite challenges, the pipeline successfully processed all 8 proteins
