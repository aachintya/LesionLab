import torch
from torch.utils.data import WeightedRandomSampler


def create_sampler(train_df, *, power: float = 1.0):
    """Weighted sampler with tunable balancing strength.

    power=1.0 fully flattens the class distribution; 0.5 is partial (sqrt)
    oversampling; 0.0 disables oversampling. Keep the combined strength of
    (sampler power + class-weight power) near 1.0 to avoid over-correcting.
    """
    class_counts = train_df["target"].value_counts().sort_index()

    sample_weights = train_df["target"].map(
        lambda x: (1.0 / class_counts[x]) ** power
    ).values.copy()

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler