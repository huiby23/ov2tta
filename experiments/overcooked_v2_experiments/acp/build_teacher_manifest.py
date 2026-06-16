from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Build ACP oracle-teacher manifest from pair-level diagnostics.')
    parser.add_argument('--diagnostics-csv', default='reports/paper_analysis_20260523/tables/pair_level_diagnostics_all_available.csv')
    parser.add_argument('--results-csv', default='reports/current_complete_results_with_diagnostics_20260523.csv')
    parser.add_argument('--method', required=True, help='Method key or display method to use as teacher pool.')
    parser.add_argument('--run-dir', default=None, help='Override teacher source run directory.')
    parser.add_argument('--backend', default=None, choices=['ppo', 'mappo'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-self-teacher', action='store_true')
    return parser.parse_args()


def _resolve_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    mask = df['method'].astype(str).eq(method)
    if 'display_method' in df.columns:
        mask = mask | df['display_method'].astype(str).eq(method)
    out = df[mask].copy()
    if out.empty:
        candidates = sorted(set(df.get('display_method', df['method']).astype(str)))
        raise SystemExit('No diagnostics rows for method %r. Known examples: %s' % (method, candidates[:20]))
    return out


def main():
    args = parse_args()
    pair_df = pd.read_csv(args.diagnostics_csv)
    pair_df = _resolve_method(pair_df, args.method)
    pair_df = pair_df[pair_df['row_type'].astype(str).eq('cross_coverage')].copy()
    if pair_df.empty:
        raise SystemExit('No cross_coverage rows found for %s' % args.method)

    run_dir = args.run_dir
    backend = args.backend
    display_method = str(pair_df['display_method'].dropna().iloc[0]) if 'display_method' in pair_df and pair_df['display_method'].notna().any() else args.method
    method_key = str(pair_df['method'].dropna().iloc[0])
    if run_dir is None or backend is None:
        res = pd.read_csv(args.results_csv)
        mask = res['method'].astype(str).eq(display_method) | res['method'].astype(str).eq(method_key)
        if not mask.any():
            raise SystemExit('Could not resolve run_dir/backend from %s for %s' % (args.results_csv, args.method))
        row = res[mask].iloc[0]
        run_dir = run_dir or row['run_dir']
        backend = backend or row['backend']

    partner_rows = []
    for partner, group in pair_df.groupby('run_j'):
        g = group.copy()
        if not args.allow_self_teacher:
            g = g[g['run_i'].astype(str) != g['run_j'].astype(str)]
        if g.empty:
            g = group.copy()
        best = g.sort_values('mean_reward', ascending=False).iloc[0]
        partner_idx = int(str(partner).split('_')[1])
        teacher_idx = int(str(best['run_i']).split('_')[1])
        partner_rows.append({
            'partner_run': partner_idx,
            'teacher_run': teacher_idx,
            'mean_reward': float(best['mean_reward']),
            'policy_tv': float(best.get('policy_tv', float('nan'))),
            'action_agreement': float(best.get('action_agreement', float('nan'))),
            'xp_out_of_sp_support_rate': float(best.get('xp_out_of_sp_support_rate', float('nan'))),
        })
    partner_rows = sorted(partner_rows, key=lambda x: x['partner_run'])
    manifest = {
        'source_method': display_method,
        'method_key': method_key,
        'source_run_dir': str(run_dir),
        'backend': backend,
        'diagnostics_csv': args.diagnostics_csv,
        'teacher_rule': 'argmax mean_reward over run_i for each run_j; self excluded unless no alternative',
        'partners': partner_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(out), 'source_method': display_method, 'run_dir': str(run_dir), 'backend': backend, 'num_partners': len(partner_rows)}, indent=2))


if __name__ == '__main__':
    main()
