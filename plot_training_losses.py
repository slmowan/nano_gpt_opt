from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List, Union, Optional


Number = Union[int, float]


def _extract_from_mapping(entry: dict, index: int) -> Optional[Tuple[Number, Number]]:
    """Extract loss and step from a mapping entry.

    Supports common key names and falls back to the entry index for the step
    when it is not provided.
    """
    loss_keys = ["loss", "value", "train_loss"]
    step_keys = ["step", "iteration", "iter", "index"]

    loss = next((entry.get(key) for key in loss_keys if key in entry), None)
    step = next((entry.get(key) for key in step_keys if key in entry), None)

    if loss is None:
        return None

    return loss, step if step is not None else index


def _extract_train_losses(train_losses: Iterable, start_index: int = 0) -> Tuple[List[Number], List[Number]]:
    """Parse the raw train_losses structure into separate loss and step lists."""
    losses: List[Number] = []
    steps: List[Number] = []

    for i, entry in enumerate(train_losses):
        idx = start_index + i

        if isinstance(entry, dict):
            parsed = _extract_from_mapping(entry, idx)
            if parsed is None:
                continue
            loss, step = parsed
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            first, second = entry
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                # Default format used elsewhere in the repo: [loss, step]
                loss, step = first, second
            else:
                continue
        elif isinstance(entry, (int, float)):
            loss, step = entry, idx
        else:
            continue

        losses.append(loss)
        steps.append(step)

    return losses, steps


def plot_training_losses(experiments_dir: Path, output_dir: Path) -> None:
    """Plot training loss curves for every JSON experiment file."""
    experiments_dir = experiments_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return

    json_files = sorted(experiments_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {experiments_dir}")
        return

    for json_file in json_files:
        with json_file.open("r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {json_file}")
                continue

        if "train_losses" not in data:
            print(f"train_losses missing in {json_file.name}, skipping.")
            continue

        losses, steps = _extract_train_losses(data["train_losses"])
        if not losses or not steps:
            print(f"No train loss data found in {json_file.name}, skipping.")
            continue

        relative_path = json_file.relative_to(experiments_dir)
        output_path = output_dir / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, linewidth=2)
        plt.xlabel("Training Steps")
        plt.ylabel("Training Loss")
        title = data.get("optimizer", relative_path.stem)
        plt.title(f"Training Loss: {title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved training loss plot to {output_path}")


if __name__ == "__main__":
    plot_training_losses(Path("experiments"), Path("training_loss_plot"))
