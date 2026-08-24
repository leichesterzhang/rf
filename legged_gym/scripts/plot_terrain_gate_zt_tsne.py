import argparse
import csv
from pathlib import Path

try:
    import matplotlib
    import numpy as np
    from sklearn.manifold import TSNE
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", "dependency")
    raise SystemExit(
        "Missing Python dependency for t-SNE plotting: "
        f"{missing_name}. Please install numpy, matplotlib, and scikit-learn "
        "in your training environment first."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize terrain gating weights and z_t distributions with t-SNE.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Directory produced by play recorder, containing terrain_gate_zt_raw.npz.",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Path to terrain_gate_zt_raw.npz. Overrides --record-dir when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots and embedding CSVs. Defaults to <record-dir>/tsne_plots.",
    )
    parser.add_argument(
        "--max-samples-per-id",
        type=int,
        default=500,
        help="Maximum number of samples kept for each terrain id before t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity. Will be clipped to a valid range automatically.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for downsampling and t-SNE.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    args = parser.parse_args()
    if args.raw_path is None and args.record_dir is None:
        parser.error("Either --record-dir or --raw-path must be provided.")
    return args


def resolve_paths(args):
    if args.raw_path is not None:
        raw_path = Path(args.raw_path).expanduser().resolve()
        record_dir = raw_path.parent
    else:
        record_dir = Path(args.record_dir).expanduser().resolve()
        raw_path = record_dir / "terrain_gate_zt_raw.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw recorder file not found: {raw_path}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = record_dir / "tsne_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_path, record_dir, output_dir


def collect_gating_data(raw_data):
    features = []
    pred_labels = []
    actual_labels = []

    for key in sorted(raw_data.files):
        if not key.startswith("gating_weights_pred_"):
            continue
        terrain_id = int(key.rsplit("_", 1)[-1])
        actual_key = f"actual_terrain_for_pred_{terrain_id}"
        if actual_key not in raw_data:
            raise KeyError(f"Missing paired key: {actual_key}")

        gating = np.asarray(raw_data[key], dtype=np.float32)
        actual = np.asarray(raw_data[actual_key], dtype=np.int64).reshape(-1)
        if gating.ndim != 2:
            gating = gating.reshape(gating.shape[0], -1)
        if gating.shape[0] != actual.shape[0]:
            raise ValueError(
                f"Sample count mismatch for predicted terrain {terrain_id}: "
                f"{gating.shape[0]} gating samples vs {actual.shape[0]} actual terrain labels"
            )
        features.append(gating)
        pred_labels.append(np.full(gating.shape[0], terrain_id, dtype=np.int64))
        actual_labels.append(actual)

    if not features:
        raise ValueError("No gating weight arrays found in the recorder file.")

    return (
        np.concatenate(features, axis=0),
        np.concatenate(pred_labels, axis=0),
        np.concatenate(actual_labels, axis=0),
    )


def collect_z_t_data(raw_data):
    features = []
    actual_labels = []

    for key in sorted(raw_data.files):
        if not key.startswith("z_t_actual_terrain_"):
            continue
        terrain_id = int(key.rsplit("_", 1)[-1])
        z_t = np.asarray(raw_data[key], dtype=np.float32)
        if z_t.ndim != 2:
            z_t = z_t.reshape(z_t.shape[0], -1)
        features.append(z_t)
        actual_labels.append(np.full(z_t.shape[0], terrain_id, dtype=np.int64))

    if not features:
        raise ValueError("No z_t arrays found in the recorder file.")

    return np.concatenate(features, axis=0), np.concatenate(actual_labels, axis=0)


def downsample_by_label(features, labels, max_samples_per_id, rng, extra_labels=None):
    keep_indices = []
    for terrain_id in np.unique(labels):
        terrain_indices = np.flatnonzero(labels == terrain_id)
        if terrain_indices.size > max_samples_per_id > 0:
            terrain_indices = rng.choice(
                terrain_indices,
                size=max_samples_per_id,
                replace=False,
            )
        keep_indices.append(np.sort(terrain_indices))
    keep_indices = np.concatenate(keep_indices, axis=0)
    sampled = [features[keep_indices], labels[keep_indices]]
    if extra_labels is not None:
        sampled.append(extra_labels[keep_indices])
    return sampled


def standardize_features(features):
    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1.0e-6, 1.0, feature_std)
    return (features - feature_mean) / feature_std


def pick_perplexity(num_samples, requested):
    if num_samples < 3:
        raise ValueError("t-SNE needs at least 3 samples.")
    upper = max(2.0, float(num_samples - 1))
    return min(float(requested), upper)


def run_tsne(features, perplexity, learning_rate, seed):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return tsne.fit_transform(features)


def make_color_map(labels):
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    return {
        int(label): cmap(idx % max(len(unique_labels), 1))
        for idx, label in enumerate(unique_labels)
    }


def annotate_centers(ax, embedding, labels):
    for terrain_id in np.unique(labels):
        mask = labels == terrain_id
        center = embedding[mask].mean(axis=0)
        ax.text(
            float(center[0]),
            float(center[1]),
            str(int(terrain_id)),
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.7},
        )


def scatter_by_label(ax, embedding, labels, title, legend_title):
    color_map = make_color_map(labels)
    unique_labels = np.unique(labels)
    for terrain_id in unique_labels:
        mask = labels == terrain_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.75,
            c=[color_map[int(terrain_id)]],
            label=str(int(terrain_id)),
            edgecolors="none",
        )
    annotate_centers(ax, embedding, labels)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(
        title=legend_title,
        loc="best",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )


def save_embedding_csv(csv_path, embedding, primary_labels, primary_name, secondary_labels=None, secondary_name=None):
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["x", "y", primary_name]
        if secondary_labels is not None and secondary_name is not None:
            header.append(secondary_name)
        writer.writerow(header)
        for idx in range(embedding.shape[0]):
            row = [
                float(embedding[idx, 0]),
                float(embedding[idx, 1]),
                int(primary_labels[idx]),
            ]
            if secondary_labels is not None:
                row.append(int(secondary_labels[idx]))
            writer.writerow(row)


def plot_and_save(
    features,
    labels,
    title,
    legend_title,
    output_png,
    output_csv,
    seed,
    perplexity,
    learning_rate,
    dpi,
    secondary_labels=None,
    secondary_name=None,
    show=False,
):
    features = standardize_features(features)
    actual_perplexity = pick_perplexity(features.shape[0], perplexity)
    embedding = run_tsne(features, actual_perplexity, learning_rate, seed)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_by_label(ax, embedding, labels, f"{title} (perplexity={actual_perplexity:.1f})", legend_title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    save_embedding_csv(
        output_csv,
        embedding,
        labels,
        legend_title,
        secondary_labels=secondary_labels,
        secondary_name=secondary_name,
    )
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    raw_path, record_dir, output_dir = resolve_paths(args)
    rng = np.random.default_rng(args.seed)

    raw_data = np.load(raw_path)

    gating_features, gating_pred_labels, gating_actual_labels = collect_gating_data(raw_data)
    gating_features, gating_pred_labels, gating_actual_labels = downsample_by_label(
        gating_features,
        gating_pred_labels,
        args.max_samples_per_id,
        rng,
        extra_labels=gating_actual_labels,
    )

    z_t_features, z_t_actual_labels = collect_z_t_data(raw_data)
    z_t_features, z_t_actual_labels = downsample_by_label(
        z_t_features,
        z_t_actual_labels,
        args.max_samples_per_id,
        rng,
    )

    print(f"Loaded recorder from: {record_dir}")
    print(
        f"Gating samples for t-SNE: {gating_features.shape[0]} "
        f"(feature dim={gating_features.shape[1]})"
    )
    print(
        f"z_t samples for t-SNE: {z_t_features.shape[0]} "
        f"(feature dim={z_t_features.shape[1]})"
    )

    gating_png = output_dir / "gating_tsne_by_predicted_terrain.png"
    gating_csv = output_dir / "gating_tsne_embedding.csv"
    plot_and_save(
        gating_features,
        gating_pred_labels,
        title="Gating Weights t-SNE",
        legend_title="pred_terrain_id",
        output_png=gating_png,
        output_csv=gating_csv,
        seed=args.seed,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        dpi=args.dpi,
        secondary_labels=gating_actual_labels,
        secondary_name="actual_terrain_id",
        show=args.show,
    )

    z_t_png = output_dir / "z_t_tsne_by_actual_terrain.png"
    z_t_csv = output_dir / "z_t_tsne_embedding.csv"
    plot_and_save(
        z_t_features,
        z_t_actual_labels,
        title="z_t t-SNE",
        legend_title="actual_terrain_id",
        output_png=z_t_png,
        output_csv=z_t_csv,
        seed=args.seed,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Saved gating t-SNE plot to: {gating_png}")
    print(f"Saved gating t-SNE embedding to: {gating_csv}")
    print(f"Saved z_t t-SNE plot to: {z_t_png}")
    print(f"Saved z_t t-SNE embedding to: {z_t_csv}")


if __name__ == "__main__":
    main()
