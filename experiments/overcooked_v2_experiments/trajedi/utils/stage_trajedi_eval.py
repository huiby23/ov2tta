import argparse
import shutil
from pathlib import Path


def copy_ckpt(src_ckpt: Path, dst_run: Path):
    if not src_ckpt.is_dir():
        raise FileNotFoundError(f"missing checkpoint: {src_ckpt}")
    dst_ckpt = dst_run / "ckpt_final"
    dst_run.mkdir(parents=True, exist_ok=True)
    if dst_ckpt.exists():
        shutil.rmtree(dst_ckpt)
    shutil.copytree(src_ckpt, dst_ckpt)


def stage(run_dir: Path, run_num: int, mode: str, output: Path | None):
    src_run = run_dir / f"run_{run_num}"
    if not src_run.is_dir():
        raise FileNotFoundError(f"missing TrajeDi run directory: {src_run}")
    if output is None:
        output = run_dir / f"eval_{mode}_run_{run_num}"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    out_idx = 0
    if mode in {"ego_partners", "ego_only"}:
        copy_ckpt(src_run / "ego" / "ckpt_final", output / f"run_{out_idx}")
        out_idx += 1
    if mode in {"partners", "ego_partners"}:
        partner_dirs = sorted(
            [p for p in src_run.iterdir() if p.is_dir() and p.name.startswith("partner_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not partner_dirs:
            raise FileNotFoundError(f"no partner_* checkpoints under {src_run}")
        for partner_dir in partner_dirs:
            copy_ckpt(partner_dir / "ckpt_final", output / f"run_{out_idx}")
            out_idx += 1
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Stage TrajeDi ego/partner checkpoints for PPO visualizer cross-play evaluation.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--run_num", type=int, default=0)
    parser.add_argument("--mode", choices=["partners", "ego_partners", "ego_only"], default="partners")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stage(args.run_dir, args.run_num, args.mode, args.output)


if __name__ == "__main__":
    main()
