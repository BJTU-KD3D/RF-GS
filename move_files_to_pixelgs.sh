#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${1:-./RF-GS}"

required=(
  "$target_dir/train.py"
  "$target_dir/scene/gaussian_model.py"
  "$target_dir/scene/dataset_readers.py"
  "$target_dir/utils/loss_utils.py"
  "$target_dir/arguments/__init__.py"
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: expected RF-GS file not found: $path" >&2
    echo "Usage: bash move_files_to_pixelgs.sh /path/to/RF-GS" >&2
    exit 1
  fi
done

backup_dir="$target_dir/.rf-gs-backup"
mkdir -p "$backup_dir/scene" "$backup_dir/utils" "$backup_dir/arguments"

backup_once() {
  local source_path="$1"
  local backup_path="$2"
  if [[ ! -f "$backup_path" ]]; then
    cp -p "$source_path" "$backup_path"
  fi
}

backup_once "$target_dir/train.py" "$backup_dir/train.py"
backup_once "$target_dir/scene/gaussian_model.py" "$backup_dir/scene/gaussian_model.py"
backup_once "$target_dir/scene/dataset_readers.py" "$backup_dir/scene/dataset_readers.py"
backup_once "$target_dir/utils/loss_utils.py" "$backup_dir/utils/loss_utils.py"
backup_once "$target_dir/arguments/__init__.py" "$backup_dir/arguments/__init__.py"
if [[ -f "$target_dir/rf_pruning.py" ]]; then
  backup_once "$target_dir/rf_pruning.py" "$backup_dir/rf_pruning.py"
fi

install -m 0644 "$script_dir/train.py" "$target_dir/train.py"
install -m 0644 "$script_dir/rf_pruning.py" "$target_dir/rf_pruning.py"
install -m 0644 "$script_dir/gaussian_model.py" "$target_dir/scene/gaussian_model.py"
install -m 0644 "$script_dir/dataset_readers.py" "$target_dir/scene/dataset_readers.py"
install -m 0644 "$script_dir/loss_utils.py" "$target_dir/utils/loss_utils.py"
install -m 0644 "$script_dir/arguments.py" "$target_dir/arguments/__init__.py"
install -m 0755 "$script_dir/run.sh" "$target_dir/run.sh"

echo "RF-GS patch installed in: $target_dir"
echo "Original files saved in: $backup_dir"
