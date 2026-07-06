#!/usr/bin/env bash
# 00_download_data.sh - pull all raw inputs. Fails loud on any non-200.
# TCGA-BRCA from UCSC Xena; METABRIC from cBioPortal datahub (git-LFS via media host).
set -euo pipefail
RAW="$(dirname "$0")/../data/raw"; mkdir -p "$RAW"

fetch () {  # url  outfile
  echo "-> $2"
  code=$(curl -sL "$1" -o "$RAW/$2" -w "%{http_code}")
  [[ "$code" == "200" ]] || { echo "FAIL: HTTP $code for $1"; exit 1; }
}

# --- TCGA-BRCA (UCSC Xena) ---
fetch "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"                       TCGA-BRCA.rnaseq.tsv.gz
fetch "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HumanMethylation450.gz"           TCGA-BRCA.methyl450.tsv.gz
fetch "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp" TCGA-BRCA.survival.tsv
fetch "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"              TCGA-BRCA.clinical.tsv
fetch "https://tcga.xenahubs.net/download/probeMap/illuminaMethyl450_hg19_GPL16304_TCGAlegacy"  methyl450_probemap.tsv
gunzip -f "$RAW/TCGA-BRCA.rnaseq.tsv.gz"    # -> TCGA-BRCA.rnaseq.tsv

# --- METABRIC (cBioPortal datahub, served via media.githubusercontent LFS) ---
MB="https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric"
fetch "$MB/data_clinical_patient.txt"           metabric_clinical_patient.txt
fetch "$MB/data_mrna_illumina_microarray.txt"   metabric_mrna.txt

echo "all downloads OK -> $RAW"
