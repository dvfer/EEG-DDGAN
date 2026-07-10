"""Utility to convert MOABB / MNE Epochs data to the long-format CSV
accepted by eeggan.helpers.dataloader.Dataloader.

Usage example
-------------
    from moabb.datasets import BNCI2014001
    from moabb.paradigms import MotorImagery
    import numpy as np
    from eeggan.helpers.moabb_export import export_to_csv

    dataset = BNCI2014001()
    paradigm = MotorImagery()
    X, y, metadata = paradigm.get_data(dataset, subjects=[1])
    # X: (n_trials, n_channels, n_timepoints)
    export_to_csv(X, labels=y, output_path='eeg_data.csv')

Then train with:
    python -m eeggan.gan_training_main \\
        data=eeg_data.csv kw_channel=Channel kw_time=Time kw_conditions=Condition
"""

import numpy as np
import pandas as pd


def export_to_csv(data, labels=None, channel_names=None, output_path='eeg_data.csv'):
    """Export EEG data to long-format CSV compatible with Dataloader.

    Args:
        data: numpy array (n_trials, n_channels, n_timepoints)
              OR MNE Epochs object.
        labels: 1-D array-like of length n_trials, optional.
                Class labels / conditions for each trial.
        channel_names: list of str, length n_channels, optional.
                       Defaults to ['Ch0', 'Ch1', ...].
        output_path: path for the output CSV file.

    Output CSV format (one row per channel per trial):
        Channel  | Condition (optional) | Time_0 | Time_1 | ... | Time_N
        ---------|----------------------|--------|--------|-----|-------
        Ch0      | 1                    | 0.12   | -0.03  | ... |
        Ch1      | 1                    | 0.05   | 0.11   | ... |
        Ch0      | 2                    | -0.08  | 0.22   | ... |
        ...

    Load with:
        Dataloader(path, kw_channel='Channel', kw_time='Time',
                   kw_conditions='Condition')    # omit kw_conditions if no labels
    """
    # Accept MNE Epochs
    try:
        import mne
        if isinstance(data, mne.Epochs):
            if channel_names is None:
                channel_names = data.ch_names
            if labels is None:
                labels = data.events[:, -1]
            data = data.get_data()  # (n_trials, n_channels, n_timepoints)
    except ImportError:
        pass

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(
            f"data must be 3-D (n_trials, n_channels, n_timepoints), got shape {data.shape}"
        )

    n_trials, n_channels, n_timepoints = data.shape

    if channel_names is None:
        channel_names = [f'Ch{i}' for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(channel_names)}) must match n_channels ({n_channels})"
        )

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != n_trials:
            raise ValueError(
                f"labels length ({len(labels)}) must match n_trials ({n_trials})"
            )

    time_cols = [f'Time_{t}' for t in range(n_timepoints)]
    rows = []
    for trial_idx in range(n_trials):
        for ch_idx, ch_name in enumerate(channel_names):
            row = {'Channel': ch_name}
            if labels is not None:
                row['Condition'] = labels[trial_idx]
            row.update(dict(zip(time_cols, data[trial_idx, ch_idx])))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(
        f"Saved {n_trials} trials × {n_channels} channels × {n_timepoints} timepoints "
        f"→ {len(df)} rows to '{output_path}'"
    )
    return df
