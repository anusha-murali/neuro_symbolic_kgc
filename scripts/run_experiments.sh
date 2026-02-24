#!/bin/bash
# scripts/run_experiments.sh
# Batch experiment runner for NeuroSymbolic KGC
# This script runs multiple experiments with different configurations

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Default values
CONFIG_FILE="config.yaml"
DATA_DIR="data/processed"
GPU_ID=0
EXPERIMENT_NAME="default"
DRY_RUN=false
PARALLEL_JOBS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --config FILE      Base config file (default: config.yaml)"
    echo "  -d, --data-dir DIR     Data directory (default: data/processed)"
    echo "  -g, --gpu ID           GPU ID to use (default: 0)"
    echo "  -n, --name NAME        Experiment name (default: timestamp)"
    echo "  -j, --jobs N           Number of parallel jobs (default: 1)"
    echo "  --dry-run              Print commands without executing"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --name embedding_sweep"
    echo "  $0 --gpu 1 --jobs 2"
    echo "  $0 --dry-run"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_command() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $1"
    else
        eval "$1"
    fi
}

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -n|--name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Generate timestamp if no experiment name provided
if [ "$EXPERIMENT_NAME" = "default" ]; then
    EXPERIMENT_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

# =============================================================================
# Create Experiment Directories
# =============================================================================

EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
RESULTS_DIR="${EXPERIMENT_DIR}/results"

run_command "mkdir -p ${LOG_DIR} ${CONFIG_DIR} ${RESULTS_DIR}"

log_info "Experiment: ${EXPERIMENT_NAME}"
log_info "Config file: ${CONFIG_FILE}"
log_info "Data directory: ${DATA_DIR}"
log_info "GPU ID: ${GPU_ID}"
log_info "Output directory: ${EXPERIMENT_DIR}"

# =============================================================================
# Define Experiment Variations
# =============================================================================

# Embedding dimension sweep
EMBEDDING_DIMS=(100 200 300 400)

# Learning rate sweep
LEARNING_RATES=(0.0005 0.001 0.002)

# Number of negatives sweep
NEGATIVES=(16 32 64 128)

# Rule influence (lambda) sweep
LAMBDA_LOGIC=(0.05 0.1 0.2 0.3)

# Temperature sweep
TEMPERATURE=(0.5 1.0 2.0)

# =============================================================================
# Generate Experiment Configs
# =============================================================================

declare -a EXPERIMENTS

# Experiment 1: Embedding dimension sweep
for dim in "${EMBEDDING_DIMS[@]}"; do
    EXP_NAME="emb_${dim}"
    EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
    
    # Create modified config
    cat > "${EXP_CONFIG}.tmp" << EOF
# Auto-generated experiment config for ${EXP_NAME}
# Base config: ${CONFIG_FILE}

model:
  name: "NeuroSymbolicKGC"
  embedding_dim: ${dim}
  lambda_logic: 0.2
  temperature: 1.0
  margin: 5.0

training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  n_epochs: 200
  n_negatives: 64
  eval_every: 5
  mixed_precision: true
  patience: 20
  warmup_epochs: 10
  max_val_batches: 50

symbolic:
  min_support: 2
  min_confidence: 0.01
  rule_types: ["inverse", "symmetric", "chain", "composition"]
  max_rules: 1000
  mine_rules: true

validation:
  fast_mode: true
  sampling:
    num_samples: 1000
    correction: true
EOF
    
    if [ "$DRY_RUN" = false ]; then
        mv "${EXP_CONFIG}.tmp" "${EXP_CONFIG}"
    fi
    
    EXPERIMENTS+=("${EXP_NAME}")
done

# Experiment 2: Learning rate sweep
for lr in "${LEARNING_RATES[@]}"; do
    EXP_NAME="lr_${lr}"
    EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
    
    # Create modified config
    cat > "${EXP_CONFIG}.tmp" << EOF
# Auto-generated experiment config for ${EXP_NAME}
# Base config: ${CONFIG_FILE}

model:
  name: "NeuroSymbolicKGC"
  embedding_dim: 200
  lambda_logic: 0.2
  temperature: 1.0
  margin: 5.0

training:
  batch_size: 512
  learning_rate: ${lr}
  weight_decay: 0.00001
  n_epochs: 200
  n_negatives: 64
  eval_every: 5
  mixed_precision: true
  patience: 20
  warmup_epochs: 10
  max_val_batches: 50

symbolic:
  min_support: 2
  min_confidence: 0.01
  rule_types: ["inverse", "symmetric", "chain", "composition"]
  max_rules: 1000
  mine_rules: true

validation:
  fast_mode: true
  sampling:
    num_samples: 1000
    correction: true
EOF
    
    if [ "$DRY_RUN" = false ]; then
        mv "${EXP_CONFIG}.tmp" "${EXP_CONFIG}"
    fi
    
    EXPERIMENTS+=("${EXP_NAME}")
done

# Experiment 3: Number of negatives sweep
for neg in "${NEGATIVES[@]}"; do
    EXP_NAME="neg_${neg}"
    EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
    
    # Create modified config
    cat > "${EXP_CONFIG}.tmp" << EOF
# Auto-generated experiment config for ${EXP_NAME}
# Base config: ${CONFIG_FILE}

model:
  name: "NeuroSymbolicKGC"
  embedding_dim: 200
  lambda_logic: 0.2
  temperature: 1.0
  margin: 5.0

training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  n_epochs: 200
  n_negatives: ${neg}
  eval_every: 5
  mixed_precision: true
  patience: 20
  warmup_epochs: 10
  max_val_batches: 50

symbolic:
  min_support: 2
  min_confidence: 0.01
  rule_types: ["inverse", "symmetric", "chain", "composition"]
  max_rules: 1000
  mine_rules: true

validation:
  fast_mode: true
  sampling:
    num_samples: 1000
    correction: true
EOF
    
    if [ "$DRY_RUN" = false ]; then
        mv "${EXP_CONFIG}.tmp" "${EXP_CONFIG}"
    fi
    
    EXPERIMENTS+=("${EXP_NAME}")
done

# Experiment 4: Rule influence (lambda) sweep
for lam in "${LAMBDA_LOGIC[@]}"; do
    EXP_NAME="lambda_${lam}"
    EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
    
    # Create modified config
    cat > "${EXP_CONFIG}.tmp" << EOF
# Auto-generated experiment config for ${EXP_NAME}
# Base config: ${CONFIG_FILE}

model:
  name: "NeuroSymbolicKGC"
  embedding_dim: 200
  lambda_logic: ${lam}
  temperature: 1.0
  margin: 5.0

training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  n_epochs: 200
  n_negatives: 64
  eval_every: 5
  mixed_precision: true
  patience: 20
  warmup_epochs: 10
  max_val_batches: 50

symbolic:
  min_support: 2
  min_confidence: 0.01
  rule_types: ["inverse", "symmetric", "chain", "composition"]
  max_rules: 1000
  mine_rules: true

validation:
  fast_mode: true
  sampling:
    num_samples: 1000
    correction: true
EOF
    
    if [ "$DRY_RUN" = false ]; then
        mv "${EXP_CONFIG}.tmp" "${EXP_CONFIG}"
    fi
    
    EXPERIMENTS+=("${EXP_NAME}")
done

# Experiment 5: Temperature sweep
for temp in "${TEMPERATURE[@]}"; do
    EXP_NAME="temp_${temp}"
    EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
    
    # Create modified config
    cat > "${EXP_CONFIG}.tmp" << EOF
# Auto-generated experiment config for ${EXP_NAME}
# Base config: ${CONFIG_FILE}

model:
  name: "NeuroSymbolicKGC"
  embedding_dim: 200
  lambda_logic: 0.2
  temperature: ${temp}
  margin: 5.0

training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  n_epochs: 200
  n_negatives: 64
  eval_every: 5
  mixed_precision: true
  patience: 20
  warmup_epochs: 10
  max_val_batches: 50

symbolic:
  min_support: 2
  min_confidence: 0.01
  rule_types: ["inverse", "symmetric", "chain", "composition"]
  max_rules: 1000
  mine_rules: true

validation:
  fast_mode: true
  sampling:
    num_samples: 1000
    correction: true
EOF
    
    if [ "$DRY_RUN" = false ]; then
        mv "${EXP_CONFIG}.tmp" "${EXP_CONFIG}"
    fi
    
    EXPERIMENTS+=("${EXP_NAME}")
done

# =============================================================================
# Run Experiments
# =============================================================================

log_info "Generated ${#EXPERIMENTS[@]} experiment configurations"
log_info "Experiments: ${EXPERIMENTS[*]}"

# Create a summary file
SUMMARY_FILE="${EXPERIMENT_DIR}/experiment_summary.txt"
{
    echo "NeuroSymbolic KGC Experiment Summary"
    echo "===================================="
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Date: $(date)"
    echo "Base config: ${CONFIG_FILE}"
    echo "Data directory: ${DATA_DIR}"
    echo "GPU ID: ${GPU_ID}"
    echo ""
    echo "Experiments:"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
} > "${SUMMARY_FILE}"

log_info "Experiment summary saved to: ${SUMMARY_FILE}"

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_config="${CONFIG_DIR}/${exp_name}.yaml"
    local exp_log="${LOG_DIR}/${exp_name}.log"
    local exp_results="${RESULTS_DIR}/${exp_name}"
    
    log_info "Starting experiment: ${exp_name}"
    
    # Create results subdirectory
    mkdir -p "${exp_results}"
    
    # Run the experiment
    CMD="python src/main.py \
        --config ${exp_config} \
        --data_dir ${DATA_DIR} \
        > ${exp_log} 2>&1"
    
    run_command "${CMD}"
    
    if [ $? -eq 0 ]; then
        log_success "Experiment ${exp_name} completed successfully"
        # Copy best model if it exists
        if [ -f "checkpoints/best_model.pt" ]; then
            run_command "cp checkpoints/best_model.pt ${exp_results}/"
        fi
        # Copy final model if it exists
        if [ -f "checkpoints/final_model.pt" ]; then
            run_command "cp checkpoints/final_model.pt ${exp_results}/"
        fi
    else
        log_error "Experiment ${exp_name} failed. Check log: ${exp_log}"
    fi
}

# Run experiments sequentially or in parallel
if [ ${PARALLEL_JOBS} -gt 1 ]; then
    log_info "Running experiments in parallel with ${PARALLEL_JOBS} jobs"
    
    # Export the function and variables for parallel execution
    export -f run_experiment
    export CONFIG_DIR LOG_DIR RESULTS_DIR DATA_DIR GPU_ID DRY_RUN
    
    # Run in parallel using xargs
    printf "%s\n" "${EXPERIMENTS[@]}" | xargs -P ${PARALLEL_JOBS} -I {} bash -c 'run_experiment "$@"' _ {}
else
    log_info "Running experiments sequentially"
    for exp in "${EXPERIMENTS[@]}"; do
        run_experiment "${exp}"
    done
fi

# =============================================================================
# Collect Results
# =============================================================================

log_info "Collecting results..."

RESULTS_FILE="${EXPERIMENT_DIR}/all_results.csv"
echo "experiment,mrr,hits1,hits3,hits10" > "${RESULTS_FILE}"

for exp in "${EXPERIMENTS[@]}"; do
    RESULT_DIR="${RESULTS_DIR}/${exp}"
    
    # Try to extract metrics from the log
    LOG_FILE="${LOG_DIR}/${exp}.log"
    if [ -f "${LOG_FILE}" ]; then
        MRR=$(grep -E "Val MRR: [0-9.]+" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        HITS1=$(grep -E "Val Hits@1: [0-9.]+" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        HITS3=$(grep -E "Val Hits@3: [0-9.]+" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        HITS10=$(grep -E "Val Hits@10: [0-9.]+" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
        
        if [ -n "${MRR}" ]; then
            echo "${exp},${MRR},${HITS1},${HITS3},${HITS10}" >> "${RESULTS_FILE}"
            log_success "Collected results for ${exp}: MRR=${MRR}"
        else
            log_warning "No metrics found for ${exp}"
        fi
    fi
done

log_success "Results saved to: ${RESULTS_FILE}"

# =============================================================================
# Generate Summary Report
# =============================================================================

REPORT_FILE="${EXPERIMENT_DIR}/report.md"

cat > "${REPORT_FILE}" << EOF
# NeuroSymbolic KGC Experiment Report

## Experiment Overview

- **Name**: ${EXPERIMENT_NAME}
- **Date**: $(date)
- **Base Config**: ${CONFIG_FILE}
- **Data Directory**: ${DATA_DIR}
- **GPU ID**: ${GPU_ID}

## Experiments Run

Total experiments: ${#EXPERIMENTS[@]}

| Experiment | MRR | Hits@1 | Hits@3 | Hits@10 |
|------------|-----|--------|--------|---------|
EOF

# Add results to report
while IFS=, read -r exp mrr hits1 hits3 hits10; do
    if [ "${exp}" != "experiment" ]; then
        echo "| ${exp} | ${mrr} | ${hits1} | ${hits3} | ${hits10} |" >> "${REPORT_FILE}"
    fi
done < "${RESULTS_FILE}"

cat >> "${REPORT_FILE}" << EOF

## Best Configuration

$(sort -t, -k2 -nr "${RESULTS_FILE}" | head -2 | tail -1 | awk -F, '{print "The best configuration was **" $1 "** with MRR = " $2}')

## Notes

- All experiments used filtered evaluation protocol
- Early stopping patience: 20 epochs
- Validation performed every 5 epochs
EOF

log_success "Report generated: ${REPORT_FILE}"

# =============================================================================
# Cleanup
# =============================================================================

log_success "Experiment batch ${EXPERIMENT_NAME} completed!"
log_info "Results saved in: ${EXPERIMENT_DIR}"

exit 0
