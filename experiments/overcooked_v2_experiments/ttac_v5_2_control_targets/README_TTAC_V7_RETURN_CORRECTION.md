# TTAC v7 Return-Correction Notes

This directory is copied from `ttac_v5_2_state_selection` and keeps old TTAC
variants intact. The v7 modes are post-hoc online adaptation modes for an
existing zero-adapter checkpoint.

## Motivation

The v5/v5.2 estimator did not behave like a reliable partner-specific policy
model: true, wrong, random, and delayed histories often produced similar update
directions. The stable positive signal looked more like local support refinement
than agreement alignment.

v7 therefore tests a narrower hypothesis:

> Online updates are useful mainly at local states where the base policy looks
> weak and the estimator target is meaningfully different from the base policy.

## Implemented Modes

- `ttac_v7_return_weighted`
- `ttac_v7_return_weighted_wrong_history`
- `ttac_v7_return_weighted_random_history`
- `ttac_v7_return_weighted_delayed_history`
- `ttac_v7_return_weighted_tv_gate`
- `ttac_v7_return_weighted_tv_gate_wrong_history`
- `ttac_v7_return_weighted_tv_gate_random_history`
- `ttac_v7_return_weighted_tv_gate_delayed_history`
- `ttac_v7_return_weighted_value_gate`
- `ttac_v7_return_weighted_value_gate_wrong_history`
- `ttac_v7_return_weighted_value_gate_random_history`
- `ttac_v7_return_weighted_value_gate_delayed_history`

## Update Weight

The v7 update weight is:

```text
return_score = latest_valid * tv_score * value_score
```

where:

- `tv_score` is high when the estimator target distribution differs from the
  base policy distribution.
- `value_score` is high when the queried state value is below the recent
  history value center.

The policy update still uses the v5 estimator target as a candidate correction,
but v7 makes the correction state-dependent rather than uniformly applying it.

## Interpretation Guardrail

Do not claim v7 is a solved partner-specific adaptation method unless true
history clearly beats wrong/random/delayed controls. If corrupted histories still
work similarly, v7 should be described as test-time local support refinement.
