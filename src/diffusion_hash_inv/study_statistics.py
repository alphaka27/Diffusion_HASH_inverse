"""Registered family aggregation; directions and membership are explicit inputs."""
import csv
from dataclasses import asdict
from pathlib import Path

from .evaluation import paired_comparison, holm_adjust


def analyze_family(registered, outcomes, *, bootstrap_seed):
    """outcomes: comparison -> seed -> (ordered target IDs, model bits, baseline bits).

    This consumes already G2-validated pairs. It cannot confer G0/G1/G2 PASS.
    """
    if not registered or set(registered) != set(outcomes):
        raise ValueError("complete preregistered family required; no post-hoc dropping")
    results = []
    for seed in (0, 1, 2):
        rows = []
        for comparison, direction in registered.items():
            if direction not in {"greater", "two-sided"} or set(outcomes[comparison]) != {0, 1, 2}:
                raise ValueError("declared direction and exactly three seeds required")
            ids, model, baseline = outcomes[comparison][seed]
            if len(set(ids)) != len(ids) or len(ids) != len(model) or len(ids) != len(baseline):
                raise ValueError("unique aligned target pairs required")
            if ids != outcomes[comparison][0][0]:
                raise ValueError("all seeds must use the same ordered target set")
            row = dict(comparison_id=comparison, seed=seed, alternative=direction,
                       **asdict(paired_comparison(model, baseline, bootstrap_seed=bootstrap_seed,
                                                 bootstrap_samples=10_000, alternative=direction)))
            rows.append(row)
        adjusted = holm_adjust({row['comparison_id']: row['mcnemar_pvalue'] for row in rows})
        for row in rows:
            row['holm_adjusted_pvalue'] = adjusted[row['comparison_id']]
            row['superiority_criterion'] = (row['alternative'] == 'greater'
                                           and row['holm_adjusted_pvalue'] < .05 and row['delta_ci95'][0] > 0)
        results.extend(rows)
    reproduction = {}
    for name, direction in registered.items():
        if direction != 'greater':
            continue
        values = [row for row in results if row['comparison_id'] == name]
        reproduction[name] = ('Strongly Reproduced' if all(row['delta_ci95'][0] > 0 for row in values)
                              else 'Reproduced' if all(row['absolute_gain'] > 0 for row in values) else 'Unstable')
    return results, reproduction


def write_statistics(directory, rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    columns = {
        'MCNEMAR_RESULTS.csv': ['comparison_id','seed','alternative','target_count','n10','n01','mcnemar_pvalue'],
        'BOOTSTRAP_RESULTS.csv': ['comparison_id','seed','absolute_gain','delta_ci95'],
        'HOLM_RESULTS.csv': ['comparison_id','seed','mcnemar_pvalue','holm_adjusted_pvalue'],
    }
    for name, fields in columns.items():
        with (directory / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
