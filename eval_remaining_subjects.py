"""Genera métricas (JSD, MMD², error de pico ERP) para cada sujeto contra su
propio held-out de test, reusando compare_subject() de compare_samples.py
(que ya genera las muestras sintéticas + ERP/PSD/PCA/t-SNE) — un config único
(el de moabb_pipeline.py / MODEL_PREFIX actual), no la matriz de ablation.

Uso:
    uv run python eval_remaining_subjects.py              # sujetos 2..10
    uv run python eval_remaining_subjects.py 4 5 6         # sujetos específicos
"""
import sys

import pandas as pd

from compare_samples import compare_subject
from moabb_pipeline import MODEL_PREFIX

RESULTS_CSV = f'ablation_results/metrics_{MODEL_PREFIX}.csv'

if __name__ == '__main__':
    subjects = [int(s) for s in sys.argv[1:]] or list(range(2, 11))

    rows = []
    for subject in subjects:
        print(f'\n{"="*55}\n  Sujeto {subject}\n{"="*55}')
        metrics = compare_subject(subject)  # usa defaults: trained_models/{MODEL_PREFIX}_sXXX.pt, etc.
        if metrics is None:
            print(f'  Aviso: sujeto {subject} omitido (falta checkpoint o test CSV).')
            continue
        rows.append({'subject': subject, **metrics})

    if not rows:
        print('\nNingún sujeto produjo métricas.')
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nTabla -> {RESULTS_CSV}')
    print(df.to_string(index=False))
