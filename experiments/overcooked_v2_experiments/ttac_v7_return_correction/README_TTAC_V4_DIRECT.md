# TTAC v3 Working Line

This directory is a preserved copy of `ttac_v2` created on 2026-06-08 for the next partner-specific alignment exploration.

Do not edit `ttac_v2` when developing v3. `ttac_v2` preserves the current AW / semantic-AW line and sweep results.

Current v2 finding to preserve:
- AW / semantic-AW can improve XP through a generic support-refinement effect.
- True-history does not cleanly beat wrong/delayed controls, so v3 should focus on partner-specific mismatch/alignment rather than only CE weight tuning.

Planned v3 focus:
- Keep support-refinement as a non-destructive auxiliary option.
- Add explicit partner-specific alignment diagnostics and update objectives in this directory only.
- Require true-history to beat wrong/delayed controls before full validation.
