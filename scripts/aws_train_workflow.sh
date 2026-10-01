#!/usr/bin/env bash
# aws_train_workflow.sh — Master Pipeline Script for AWS GPU EC2 Execution
# Runs License Audit -> Direct Dataset Downloads -> Dataset Audit -> Manifest Generation -> GPU Smoke Training -> STOP & Report

set -e

# Load python environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "========================================================================"
echo "      AWS Master Training Workflow — Local Indian STT"
echo "========================================================================"

# Phase 1: GPU & Disk Safety Audit
echo -e "\n[Phase 1] Auditing AWS Hardware & Disk Space..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi
df -h /

# Phase 2: License Audit Report
echo -e "\n[Phase 2] Generating Dataset License Audit Report..."
python3 -m src.datasets license-audit

# Phase 3: Direct Dataset Acquisition on AWS (Sequential with Disk Checks)
echo -e "\n[Phase 3] Acquiring Datasets Directly to AWS (data/raw/...)..."
python3 -m src.datasets download --dataset mucs-hindi-english || echo "⚠️ MUCS download skipped/failed"
python3 -m src.datasets download --dataset mozilla-cv-hindi || echo "⚠️ Common Voice download skipped/failed"
python3 -m src.datasets download --dataset vaani || echo "⚠️ Vaani download skipped/failed"

# Phase 4: Dataset Audit
echo -e "\n[Phase 4] Running Dataset Audit..."
python3 -m src.training audit

# Phase 5: Unified Manifest Generator
echo -e "\n[Phase 5] Generating Unified Manifests with Benchmark Protection..."
python3 -m src.training manifest

# Phase 6: GPU Smoke Training
echo -e "\n[Phase 6] Running GPU Smoke Training (20 Steps)..."
python3 -m src.training smoke --steps 20

# Phase 7: Comprehensive Summary Report & STOP
echo -e "\n========================================================================"
echo "          AWS GPU SETUP & PREPARATION WORKFLOW COMPLETE"
echo "========================================================================"
echo "Summary Reports Saved:"
echo "  - License Audit: data/audit/license_audit.md"
echo "  - Dataset Audit: data/audit/dataset_audit.md"
echo "  - Manifests:     data/manifests/{train.jsonl, validation.jsonl, test.jsonl}"
echo ""
echo "CRITICAL STOP POINT: Review dataset statistics above before starting full training."
echo "To launch full GPU training after approval, run:"
echo "  python3 -m src.training run --config config/training.yaml --steps 10000"
echo "========================================================================"
