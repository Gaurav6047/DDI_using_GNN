# ============================================================
# DeepDDI Final Research-Grade Streamlit Application (FINAL)
# Real GATv2 Siamese Inference + 3D Visualization +
# Confusion Matrix + t-SNE + Description Mapping + CSV Viewer
# ============================================================

import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
import py3Dmol
from stmol import showmol

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, LayerNorm

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="DeepDDI Research Platform", layout="wide")

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD METADATA FILES
# ============================================================
label_map = None
inv_label_map = None
description_map = None
report_df = None

if os.path.exists("label_map.pkl"):
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}

if os.path.exists("description_map.pkl"):
    with open("description_map.pkl", "rb") as f:
        description_map = pickle.load(f)

if os.path.exists("full_ddi_report.csv"):
    report_df = pd.read_csv("full_ddi_report.csv")

NUM_CLASSES = len(label_map) if label_map else 76

# ============================================================
# MOLECULAR GRAPH FEATURES
# ============================================================

def get_atom_features(atom):
    hyb_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4,
    }
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        hyb_map.get(atom.GetHybridization(), 5),
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        atom.GetNumRadicalElectrons(),
        int(atom.GetChiralTag()),
    ]


def get_bond_features(bond):
    bt_map = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3,
    }
    return [
        bt_map.get(bond.GetBondType(), 4),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        feat = get_bond_features(bond)
        edge_attr += [feat, feat]

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

# ============================================================
# MODEL
# ============================================================

class MedicalGNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GATv2Conv(8, 128, heads=4, edge_dim=3)
        self.n1 = LayerNorm(128 * 4)
        self.c2 = GATv2Conv(128 * 4, 128, heads=2, edge_dim=3)
        self.n2 = LayerNorm(128 * 2)
        self.c3 = GATv2Conv(128 * 2, 128, heads=1, edge_dim=3)
        self.n3 = LayerNorm(128)
        self.dropout = nn.Dropout(0.25)

    def forward(self, d):
        x, ei, ea, b = d.x, d.edge_index, d.edge_attr, d.batch
        x = F.elu(self.n1(self.c1(x, ei, ea)))
        x = self.dropout(x)
        x = F.elu(self.n2(self.c2(x, ei, ea)))
        x = self.dropout(x)
        x = F.elu(self.n3(self.c3(x, ei, ea)))
        return torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=1)


class SiameseDDI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc = MedicalGNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, g1, g2):
        return self.fc(torch.cat([self.enc(g1), self.enc(g2)], dim=1))

# ============================================================
# LOAD MODEL
# ============================================================

model = None
if os.path.exists("best_ddi_model.pt"):
    model = SiameseDDI(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("best_ddi_model.pt", map_location=device))
    model.eval()
    st.sidebar.success("✅ Trained Model Loaded")
else:
    st.sidebar.error("❌ best_ddi_model.pt not found")

# ============================================================
# 3D VISUALIZATION
# ============================================================

def generate_3d_view(smiles, width=420, height=360):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        molblock = Chem.MolToMolBlock(mol)

        view = py3Dmol.view(width=width, height=height)
        view.addModel(molblock, "mol")

        view.setStyle({
            "stick": {"radius": 0.18, "color": "gray"},
            "sphere": {"scale": 0.28}
        })

        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            view.addLabel(
                f"{atom.GetSymbol()}{idx}",
                {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "backgroundColor": "white",
                    "fontColor": "black",
                    "fontSize": 12
                }
            )

        view.zoomTo()
        return view

    except Exception:
        return None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### Test Drug Pairs")
case_option = st.sidebar.selectbox(
    "Select Test Case",
    ["Custom Input", "Aspirin + Warfarin", "Ibuprofen + Lithium", "Paracetamol + Alcohol"]
)

if case_option == "Aspirin + Warfarin":
    default_name_a, default_smiles_a = "Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"
    default_name_b, default_smiles_b = "Warfarin", "CC(=O)C(C1=CC=CC=C1)C(C2=CC=CC=C2)O"
elif case_option == "Ibuprofen + Lithium":
    default_name_a, default_smiles_a = "Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    default_name_b, default_smiles_b = "Lithium", "[Li+]"
elif case_option == "Paracetamol + Alcohol":
    default_name_a, default_smiles_a = "Paracetamol", "CC(=O)NC1=CC=C(O)C=C1O"
    default_name_b, default_smiles_b = "Ethanol", "CCO"
else:
    default_name_a = default_smiles_a = default_name_b = default_smiles_b = ""

page_mode = st.sidebar.radio("Navigate", ["Inference", "Confusion Matrix", "t-SNE", "Full Classification Report"])

# ============================================================
# MAIN UI
# ============================================================

st.title("DeepDDI Research Inference System")
st.write("Graph neural network based clinical drug-drug interaction prediction platform.")
st.divider()

# ============================================================
# INFERENCE PAGE
# ============================================================

if page_mode == "Inference":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drug A")
        name_a = st.text_input("Name A", value=default_name_a)
        smiles_a = st.text_area("SMILES A", value=default_smiles_a, height=90)
        if smiles_a:
            view = generate_3d_view(smiles_a)
            if view:
                showmol(view, height=360, width=420)

    with col2:
        st.subheader("Drug B")
        name_b = st.text_input("Name B", value=default_name_b)
        smiles_b = st.text_area("SMILES B", value=default_smiles_b, height=90)
        if smiles_b:
            view = generate_3d_view(smiles_b)
            if view:
                showmol(view, height=360, width=420)

    st.divider()

    if st.button("Run GATv2 Interaction Prediction", use_container_width=True):

        if model is None:
            st.error("Model not loaded properly.")
            st.stop()

        if not smiles_a or not smiles_b:
            st.error("Both SMILES inputs are required.")
            st.stop()

        g1 = smiles_to_graph(smiles_a).to(device)
        g2 = smiles_to_graph(smiles_b).to(device)

        with torch.no_grad():
            logits = model(g1, g2)
            probs = torch.softmax(logits, dim=1)

            pred_class = int(torch.argmax(probs, dim=1))
            confidence = float(torch.max(probs).item())

            # ✅ TOP-5
            topk_vals, topk_idxs = torch.topk(probs, k=5)

        topk_vals = topk_vals.cpu().numpy().flatten()
        topk_idxs = topk_idxs.cpu().numpy().flatten()

        top5_rows = []
        for rank, (cls_id, prob) in enumerate(zip(topk_idxs, topk_vals), start=1):

            cls_id = int(cls_id)

            if inv_label_map and cls_id in inv_label_map:
                orig_cls = inv_label_map[cls_id]
            else:
                orig_cls = None

            if description_map and orig_cls in description_map:
                short_desc = description_map[orig_cls]

                # ✅ SAME REPLACEMENT AS MAIN OUTPUT
                short_desc = short_desc.replace("#Drug1", name_a)
                short_desc = short_desc.replace("#Drug2", name_b)

                short_desc = short_desc[:90] + "..."
            else:
                short_desc = "Description not available"


            top5_rows.append({
                "Rank": rank,
                "Class ID": cls_id,
                "Interaction Description": short_desc,
                "Probability (%)": round(float(prob) * 100, 2)
            })

        top5_df = pd.DataFrame(top5_rows)

        # ✅ MAIN PREDICTION DESCRIPTION
        if inv_label_map and pred_class in inv_label_map:
            orig_class = inv_label_map[pred_class]
        else:
            orig_class = None

        if description_map and orig_class in description_map:
            description = description_map[orig_class]
            description = description.replace("#Drug1", name_a)
            description = description.replace("#Drug2", name_b)
        else:
            description = "Clinical description not available for this class."

        st.markdown(f"""
        <div style="background:white;padding:2rem;border-left:6px solid #374151;border-radius:10px;">
        <b>Predicted Interaction Class:</b> {pred_class} <br><br>
        <b>Model Confidence:</b> {confidence*100:.2f}% <br>
        <b>Inference Device:</b> {str(device).upper()} <br><br>
        <b>Clinical Interpretation:</b><br>{description}
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🔝 Top-5 Predicted Interaction Classes")
        st.dataframe(top5_df, use_container_width=True)

        st.subheader("📊 Prediction Confidence Distribution")
        fig, ax = plt.subplots()
        ax.bar(top5_df["Rank"].astype(str), top5_df["Probability (%)"])
        ax.set_xlabel("Top-5 Rank")
        ax.set_ylabel("Probability (%)")
        ax.set_title("Top-5 Interaction Prediction Confidence")
        st.pyplot(fig)

# ============================================================
# OTHER PAGES
# ============================================================

elif page_mode == "Confusion Matrix":
    if os.path.exists("Analysis_ConfusionMatrix.png"):
        st.image("Analysis_ConfusionMatrix.png", use_column_width=True)

elif page_mode == "t-SNE":
    if os.path.exists("Analysis_tSNE.png"):
        st.image("Analysis_tSNE.png", use_column_width=True)

elif page_mode == "Full Classification Report":
    if report_df is not None:
        st.dataframe(report_df, use_container_width=True)

st.markdown("<br><center><small>DeepDDI Research Platform | PyTorch | RDKit | Graph Neural Networks</small></center>", unsafe_allow_html=True)
