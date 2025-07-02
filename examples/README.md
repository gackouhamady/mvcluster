# Arabidopsis Multi-View Dataset Preparation

This script prepares a custom multi-view dataset for machine learning tasks using various biological and environmental data sources related to the Arabidopsis plant.

## 📌 Script: `prepare_custom_dataset.py`

### 🔧 Command to Run

```bash
python prepare_custom_dataset.py \
  --views Altitude_Cluster.csv Ecotype.csv Metabolomics_Rosettes.csv Metabolomics_Stems.csv \
         Phenomics_Rosettes.csv Phenomics_Stems.csv Proteomics_Rosettes_CW.csv Proteomics_Stems_CW.csv \
         Temperature.csv Transcriptomics_Rosettes.csv Transcriptomics_Rosettes_CW.csv \
         Transcriptomics_Stems.csv Transcriptomics_Stems_CW.csv \
  --labels Genetic_Cluster.csv \
  --data_name arabidopsis_all_views
```

---

# 🤮 ALOI Multi-View Dataset Preparation

This script prepares a custom multi-view dataset for machine learning tasks using different visual feature representations extracted from the ALOI (Amsterdam Library of Object Images) dataset.

## 📌 Script: `prepare_custom_dataset.py`

### 🔧 Command to Run

```bash
python prepare_custom_dataset.py \
  --views aloi-27d_clean.csv aloi-64d_clean.csv aloi-haralick-1_clean.csv aloi-hsb-3x3x3_clean.csv \
  --labels objs_labels_clean.csv \
  --data_name aloi_all_views
```

---

