"""Module `models/covariate_encoding.py`."""
import torch
import torch.nn as nn
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CovEncoder: Covariate Encoder
# Purpose:
# 1) Build perturbation / cell-type / batch encoding branches from config
# 2) Concatenate branch representations and project to a unified output dim
# ============================================================
class CovEncoder(nn.Module):
    """Covencoder implementation used by the PerturbDiff pipeline."""
    # ---------------------------
    # Initialization entry: assemble all encoding branches
    # ---------------------------
    def __init__(self, cov_cfg):
        """Special method `__init__`."""
        super().__init__()
        self.cov_cfg = cov_cfg
        hidden_dim = 0
        # Main perturbation branch (with optional Tahoe / Replogle sub-branches)
        hidden_dim += self._init_generic_perturbation_encoder(cov_cfg)
        # Cell-type branch
        hidden_dim += self._init_celltype_encoder(cov_cfg)
        # Batch branch
        hidden_dim += self._init_batch_encoder(cov_cfg)
        # Gather layer: linear projection after concatenation
        self._init_gather_layers(hidden_dim)

    # ---------------------------
    # Main perturbation branch
    # ---------------------------
    def _init_generic_perturbation_encoder(self, cov_cfg):
        """
        Generic perturbation encoder:
        - onehot/non: default categorical perturbation pathway.
        - esm2: optional PBMC-style pretrained perturbation embeddings.
        """
        hidden_dim = 0
        if cov_cfg.pert_encoding == "onehot" or cov_cfg.pert_encoding == "non":
            # Base onehot perturbation embedding (+1 reserves index 0 for no-perturbation)
            if cov_cfg.drug_encoding == "onehot" and cov_cfg.replogle_gene_encoding == "onehot":
                self.pert_encoder = nn.Embedding(
                    num_embeddings=cov_cfg.num_pert + 1,  # RNA-seq has no perturbation
                    embedding_dim=cov_cfg.hidden_dim,
                )
                hidden_dim += cov_cfg.hidden_dim
            # Optional Tahoe drug branch / Replogle gene branch
            hidden_dim += self._init_tahoe_drug_encoder(cov_cfg)
            hidden_dim += self._init_replogle_gene_encoder(cov_cfg)
        elif cov_cfg.pert_encoding == "esm2":
            # esm2 path only supports the PBMC-like configuration
            assert cov_cfg.drug_encoding == "onehot" and cov_cfg.replogle_gene_encoding == "onehot", (
                "pert_encoding=esm2 is only supported for the PBMC-like path "
                "(drug_encoding=onehot and replogle_gene_encoding=onehot)."
            )
            assert cov_cfg.pert_embedding_path is not None
            # Load pretrained embeddings and wrap as frozen embedding
            with open(cov_cfg.pert_embedding_path, "rb") as f:
                emb = pickle.load(f)
                emb = torch.tensor(list(emb.values())).float()
            emb = torch.cat([torch.zeros(1, emb.size(1)), emb], dim=0)
            emb = nn.Embedding.from_pretrained(emb, freeze=True)
            # index 79 corresponds to the control perturbation in the PBMC dataset, 
            # which is not included in the pretrained embeddings. We initialize 
            # its embedding as the mean of all other perturbation embeddings.
            emb.weight[79] += emb.weight.mean()
            self.pert_encoder = nn.Sequential(emb, nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim))
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Tahoe drug branch
    # ---------------------------
    def _init_tahoe_drug_encoder(self, cov_cfg):
        """
        Tahoe drug branch:
        - drug_encoding=chemberta_cls builds (drug_encoder + dose_encoder).
        - leaves generic pert encoder unchanged for other branches.
        """
        hidden_dim = 0
        if cov_cfg.drug_encoding == "onehot":
            pass
        elif cov_cfg.drug_encoding == "chemberta_cls":
            # Load drug embeddings (i.e., ChemBERTa CLS representations)
            with open(cov_cfg.drug_embedding_path, "rb") as fin:
                drug_embeddings = pickle.load(fin)

            # Collect and discretize doses, then build dose -> index mapping
            dose_idx = []
            for drugname_drugconc, idx in cov_cfg.pert_dict.items():
                drug, dose, _ = eval(drugname_drugconc)[0]
                dose_idx.append(dose)
            dose_idx = sorted(list(set(dose_idx)))
            dose_idx = {v: i for i, v in enumerate(dose_idx)}

            # Parse drug embedding and dose index for each perturbation entry
            all_drug_embed, all_drug_dose = {}, {}
            drug_embed_dim = None
            for drugname_drugconc, idx in cov_cfg.pert_dict.items():
                drug, dose, _ = eval(drugname_drugconc)[0]
                all_drug_dose[idx] = dose_idx[dose]
                drug = drug.strip()
                if drug == "DMSO_TF":
                    assert dose == 0.0
                    control_idx = idx
                    continue
                all_drug_embed[idx] = drug_embeddings[drug]
                drug_embed_dim = len(drug_embeddings[drug])

            # Dose encoder
            self.dose_encoder = nn.Embedding(
                num_embeddings=len(dose_idx),
                embedding_dim=cov_cfg.hidden_dim,
            )
            idx = np.zeros(len(cov_cfg.pert_dict))
            for k, v in all_drug_dose.items():
                idx[k] = v
            # Register as buffer so it moves with the model but is not trainable
            self.register_buffer("dose_indices", torch.tensor(idx, dtype=torch.int))
            hidden_dim += cov_cfg.hidden_dim

            # Drug embedding table (control entry is filled with mean embedding)
            assert sorted(all_drug_embed.keys()) == [x for x in range(len(cov_cfg.pert_dict)) if x != control_idx]
            emb = np.zeros((len(cov_cfg.pert_dict), drug_embed_dim))
            for k, v in all_drug_embed.items():
                emb[k] = v
            emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=True)
            emb.weight[control_idx] = emb.weight.mean()
            self.drug_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Replogle gene branch
    # ---------------------------
    def _init_replogle_gene_encoder(self, cov_cfg):
        """
        Replogle gene branch:
        - replogle_gene_encoding=genept replaces self.pert_encoder
          with a pretrained gene embedding encoder.
        """
        hidden_dim = 0
        if cov_cfg.replogle_gene_encoding == "onehot":
            pass
        elif cov_cfg.replogle_gene_encoding == "genept":
            # Load pretrained gene embeddings
            with open(cov_cfg.replogle_gene_embedding_path, "rb") as fin:
                rep_gene_embeddings = pickle.load(fin)
            if len(rep_gene_embeddings) == 0:
                raise ValueError(
                    f"Empty replogle gene embedding dict: {cov_cfg.replogle_gene_embedding_path}"
                )

            # Build gene embedding table; missing perturbations fall back to zero vectors.
            all_gene_embed = {}
            missing_gene_perts = []
            gene_embed_dim = len(next(iter(rep_gene_embeddings.values())))
            control_idx = None
            for gene_pert, idx in cov_cfg.pert_dict.items():
                gene_pert = str(gene_pert)
                if gene_pert == "non-targeting":
                    control_idx = idx
                    continue
                emb_vec = rep_gene_embeddings.get(gene_pert)
                if emb_vec is None:
                    all_gene_embed[idx] = np.zeros(gene_embed_dim, dtype=np.float32)
                    missing_gene_perts.append(gene_pert)
                else:
                    all_gene_embed[idx] = emb_vec

            if missing_gene_perts:
                logger.warning(
                    "Missing %d replogle perturbation embeddings (examples: %s). Using zero vectors as fallback.",
                    len(missing_gene_perts),
                    ", ".join(sorted(set(missing_gene_perts))[:10]),
                )

            emb = np.zeros((len(cov_cfg.pert_dict), gene_embed_dim))
            for k, v in all_gene_embed.items():
                emb[k] = v
            emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=True)
            if control_idx is not None:
                emb.weight[control_idx] = emb.weight.mean()
            self.pert_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Cell-type branch
    # ---------------------------
    def _init_celltype_encoder(self, cov_cfg):
        """
        Init celltype encoder.

        :param cov_cfg: Covariate encoder configuration.
        :return: Computed output(s) for this function.
        """
        hidden_dim = 0
        if cov_cfg.celltype_encoding == "onehot":
            self.celltype_encoder = nn.Embedding(
                num_embeddings=cov_cfg.num_celltype,
                embedding_dim=cov_cfg.hidden_dim,
            )
            hidden_dim += cov_cfg.hidden_dim
        elif cov_cfg.celltype_encoding == "llm":
            # Use external LLM/pretrained cell-type embeddings
            assert cov_cfg.celltype_embedding_path is not None
            with open(cov_cfg.celltype_embedding_path, "rb") as f:
                celltype_emb_dict = pickle.load(f)
            self.celltype_idx_dict = cov_cfg.cell_type_dict
            # Align indices with cell_type_dict to avoid index mismatch
            emb = {cov_cfg.cell_type_dict[k]: celltype_emb_dict[k] for k in cov_cfg.cell_type_dict}
            emb = {k: emb[k] for k in sorted(emb.keys())}
            emb = torch.tensor(list(emb.values())).float()
            emb = nn.Embedding.from_pretrained(emb, freeze=True)
            self.celltype_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Batch branch
    # ---------------------------
    def _init_batch_encoder(self, cov_cfg):
        """Execute `_init_batch_encoder` and return values used by downstream logic."""
        hidden_dim = 0
        if cov_cfg.batch_encoding is None:
            return hidden_dim
        if cov_cfg.batch_encoding == "onehot":
            self.batch_encoder = nn.Embedding(
                num_embeddings=cov_cfg.num_batch,
                embedding_dim=cov_cfg.hidden_dim,
            )
            hidden_dim += cov_cfg.hidden_dim
        return hidden_dim

    # ---------------------------
    # Gather layer
    # ---------------------------
    def _init_gather_layers(self, hidden_dim):
        # Map concatenated multi-branch representation to output dimension
        """Execute `_init_gather_layers` and return values used by downstream logic."""
        self.transform = nn.Linear(hidden_dim, self.cov_cfg.output_dim)

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(self, pert_input, celltype_input, batch_input):
        """
        Run the module forward pass.

        :param pert_input: Input `pert_input` value.
        :param celltype_input: Input `celltype_input` value.
        :param batch_input: Input `batch_input` value.
        :return: Model output tensor(s) for the given inputs.
        """
        reprs = []

        # 1) Perturbation-related representation
        if self.cov_cfg.drug_encoding == "chemberta_cls":
            # Tahoe mode: ChemBERTa drug embedding + dose embedding
            reprs.append(self.drug_encoder(pert_input))
            reprs.append(self.dose_encoder(self.dose_indices[pert_input]))
        elif self.cov_cfg.replogle_gene_encoding == "genept":
            # Replogle mode: gene embedding
            reprs.append(self.pert_encoder(pert_input))
        elif self.cov_cfg.pert_encoding == "non":
            # "non" mode: force zero index (useful for pretraining on marginal cells)
            reprs.append(self.pert_encoder(torch.zeros_like(pert_input, dtype=pert_input.dtype)))
        else:
            # onehot mode: input index +1 (index 0 is reserved for the control type)
            # or PBMC mode: cytokine embeddings (ESM2)
            reprs.append(self.pert_encoder(pert_input + 1))

        # 2) Cell-type representation
        reprs.append(self.celltype_encoder(celltype_input))

        # 3) Batch representation (optional)
        if hasattr(self, "batch_encoder"):            
            reprs.append(self.batch_encoder(batch_input))
        
        # 4) Concatenate and linearly project to output space
        reprs = torch.cat(reprs, dim=-1)
        return self.transform(reprs)
