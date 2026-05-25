#!/usr/bin/env bash
set -u

TRAIN_LOG="/workspace/shd_paper_sparse_train.log"
MONITOR_LOG="/workspace/shd_sparse_monitor.log"
APP_DIR="/app"
BASELINE_SPIKES="27734.43"
ACCEPT_SPIKES="19414.10"

TARGET_RATE="6.0"
RATE_LAM="0.005"
SPIKE_LAM="0.0"
BATCH_SIZE="128"
ENERGY_BATCH_SIZE="128"
LR="0.001"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >> "$MONITOR_LOG"
}

is_running() {
  pgrep -af 'python.*train_shd_paper.py' | grep -v grep >/dev/null 2>&1
}

latest_epoch_line() {
  grep -E 'Epoch [0-9]+ complete' "$TRAIN_LOG" 2>/dev/null | tail -n 1
}

latest_completed_epoch() {
  latest_epoch_line | sed -E 's/.*Epoch ([0-9]+) complete.*/\1/'
}

metric_from_line() {
  local label="$1"
  local line="$2"
  printf '%s\n' "$line" | sed -E "s/.*${label}: ([0-9.]+)%.*/\\1/"
}

hidden_from_line() {
  local line="$1"
  printf '%s\n' "$line" | sed -E 's/.*Hidden Hz: ([0-9.]+).*/\1/'
}

best_from_line() {
  local line="$1"
  printf '%s\n' "$line" | sed -E 's/.*Best: ([0-9.]+)%.*/\1/'
}

float_lt() {
  awk "BEGIN {exit !($1 < $2)}"
}

float_gt() {
  awk "BEGIN {exit !($1 > $2)}"
}

float_ge() {
  awk "BEGIN {exit !($1 >= $2)}"
}

float_between_ge_lt() {
  awk "BEGIN {exit !(($1 >= $2) && ($1 < $3))}"
}

start_run() {
  log "Starting SHD sparse config target_rate=${TARGET_RATE} rate_lam=${RATE_LAM} spike_lam=${SPIKE_LAM} batch_size=${BATCH_SIZE} lr=${LR}"
  cd "$APP_DIR" || exit 1
  setsid python -u train_shd_paper.py \
    --model_type sparse \
    --n_frames 1400 \
    --net_dt 0.001 \
    --tau_mem 0.02 \
    --tau_syn 0.005 \
    --epochs 60 \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dataset_path ./data \
    --output_path /workspace/new_test_results \
    --num_workers 4 \
    --paper_energy_mJ 0.42 \
    --print_every 10 \
    --validate_every 1 \
    --energy_batch_size "$ENERGY_BATCH_SIZE" \
    --target_rate "$TARGET_RATE" \
    --rate_lam "$RATE_LAM" \
    --spike_lam "$SPIKE_LAM" \
    --early_stop_patience 10 \
    --min_delta 0.002 \
    >> "$TRAIN_LOG" 2>&1 < /dev/null &
}

restart_with_config() {
  local reason="$1"
  TARGET_RATE="$2"
  RATE_LAM="$3"
  SPIKE_LAM="$4"
  BATCH_SIZE="${5:-128}"
  ENERGY_BATCH_SIZE="$BATCH_SIZE"
  log "Intervention: ${reason}. Restarting with target_rate=${TARGET_RATE} rate_lam=${RATE_LAM} spike_lam=${SPIKE_LAM} batch_size=${BATCH_SIZE}"
  printf '\n--- monitor intervention: %s target_rate=%s rate_lam=%s spike_lam=%s batch_size=%s at %s ---\n' \
    "$reason" "$TARGET_RATE" "$RATE_LAM" "$SPIKE_LAM" "$BATCH_SIZE" "$(date -Is)" >> "$TRAIN_LOG"
  pkill -f 'python.*train_shd_paper.py' >/dev/null 2>&1 || true
  sleep 10
  start_run
}

copy_best_snapshot() {
  local src="/workspace/new_test_results/shd/sparse/models/best_sparse_T1400_dt0.001.pth"
  if [ -f "$src" ]; then
    cp "$src" "/workspace/new_test_results/shd/sparse/models/best_sparse_latest_monitor_snapshot.pth" 2>/dev/null || true
  fi
}

log "Pod-side SHD sparse monitor started. interval=20m baseline_spikes=${BASELINE_SPIKES} accept_spikes=${ACCEPT_SPIKES}"

while true; do
  line="$(latest_epoch_line)"
  if [ -n "$line" ]; then
    epoch="$(latest_completed_epoch)"
    train_acc="$(metric_from_line 'Train Acc' "$line")"
    test_acc="$(metric_from_line 'Test Acc' "$line")"
    hidden_hz="$(hidden_from_line "$line")"
    best_acc="$(best_from_line "$line")"
    log "Status: epoch=${epoch} train_acc=${train_acc}% test_acc=${test_acc}% best=${best_acc}% hidden_hz=${hidden_hz}"
  else
    epoch=""
    train_acc=""
    test_acc=""
    hidden_hz=""
    best_acc=""
    log "Status: no completed epoch found yet"
  fi

  if grep -Eqi 'CUDA out of memory|out of memory' "$TRAIN_LOG" 2>/dev/null; then
    restart_with_config "CUDA OOM detected" "$TARGET_RATE" "$RATE_LAM" "$SPIKE_LAM" "64"
    sleep 1200
    continue
  fi

  if grep -Eqi '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)|Loss: (nan|inf)' "$TRAIN_LOG" 2>/dev/null; then
    LR="0.0005"
    restart_with_config "NaN/inf detected, lowering lr" "$TARGET_RATE" "$RATE_LAM" "$SPIKE_LAM" "$BATCH_SIZE"
    sleep 1200
    continue
  fi

  if ! is_running; then
    if grep -q 'Saved energy estimate to' "$TRAIN_LOG" 2>/dev/null; then
      final_acc="$(grep -E 'Final Test Accuracy:' "$TRAIN_LOG" | tail -n 1 | sed -E 's/.*Final Test Accuracy: ([0-9.]+)%.*/\1/')"
      avg_spikes="$(grep -E 'avg_spikes_per_sample:' "$TRAIN_LOG" | tail -n 1 | sed -E 's/.*avg_spikes_per_sample: ([0-9.]+).*/\1/')"
      energy_json="$(grep -E 'Saved energy estimate to' "$TRAIN_LOG" | tail -n 1 | sed -E 's/.*Saved energy estimate to //')"
      log "Completed: final_acc=${final_acc}% avg_spikes_per_sample=${avg_spikes} energy_json=${energy_json}"
      copy_best_snapshot

      if float_ge "$final_acc" "68" && float_lt "$final_acc" "78" && float_lt "$avg_spikes" "$ACCEPT_SPIKES"; then
        log "Accepted sparse model: final_acc=${final_acc}% avg_spikes=${avg_spikes}."
        exit 0
      fi

      if float_ge "$final_acc" "78"; then
        restart_with_config "final accuracy high or spikes insufficient; making sparse pressure stronger" "4.0" "0.01" "0.0" "128"
      else
        restart_with_config "final accuracy too low; relaxing sparse pressure" "8.0" "0.003" "0.0" "128"
      fi
    else
      log "Training process not running before completion; restarting active config"
      start_run
    fi

    sleep 1200
    continue
  fi

  if [ -n "$epoch" ] && [ -n "$test_acc" ] && [ -n "$hidden_hz" ]; then
    if [ "$epoch" -ge 6 ] && float_lt "$test_acc" "55" && float_lt "$hidden_hz" "4"; then
      restart_with_config "too sparse and failing after epoch ${epoch}" "8.0" "0.003" "0.0" "128"
    elif [ "$epoch" -ge 8 ] && float_ge "$test_acc" "68" && float_gt "$hidden_hz" "10"; then
      restart_with_config "healthy accuracy but hidden rate too high after epoch ${epoch}" "4.0" "0.01" "0.0" "128"
    elif [ "$epoch" -ge 10 ] && float_lt "$test_acc" "60" && float_between_ge_lt "$hidden_hz" "4" "8"; then
      restart_with_config "accuracy weak with moderate sparsity after epoch ${epoch}" "8.0" "0.005" "0.0" "128"
    fi
  fi

  sleep 1200
done
