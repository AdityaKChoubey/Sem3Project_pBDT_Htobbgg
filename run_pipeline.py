"""
run_pipeline.py
===============
Master orchestration script to execute analysis pipeline stages.

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage audit
  python run_pipeline.py --stage train
  python run_pipeline.py --stage inference
  python run_pipeline.py --stage sculpting
  python run_pipeline.py --stage correlations
  python run_pipeline.py --stage features
  python run_pipeline.py --stage ams
"""

import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd_list):
    cmd_str = " ".join(cmd_list)
    print("\n" + "=" * 78)
    print(f"[PIPELINE EXECUTE] {cmd_str}")
    print("=" * 78)
    res = subprocess.run(cmd_list, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"[PIPELINE ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline runner for WH -> bb gamma gamma analysis."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "train", "inference", "sculpting", "correlations", "features", "ams"],
        default="all",
        help="Stage to execute"
    )
    parser.add_argument("--trees", type=int, default=10, help="Number of trees for training")
    parser.add_argument("--bdt_cut", type=float, default=0.7, help="BDT score cut threshold")
    args = parser.parse_args()

    py = sys.executable

    # 1. Dataset & Weights Audit
    if args.stage in ["all", "audit"]:
        run_cmd([py, os.path.join(BASE_DIR, "01_dataset_and_weights", "check_dataset_and_weights.py")])

    # 2. Training
    if args.stage in ["all", "train"]:
        run_cmd([py, os.path.join(BASE_DIR, "02_model_training", "train_baseline_pbdt.py"), "--trees", str(args.trees)])
        run_cmd([py, os.path.join(BASE_DIR, "02_model_training", "inspect_trees.py")])

    # 3. Inference
    if args.stage in ["all", "inference"]:
        run_cmd([py, os.path.join(BASE_DIR, "03_inference_and_sculpting", "run_bdt_inference.py"), "--bdt_cut", str(args.bdt_cut)])

    # 4. Sculpting & Diagnostics
    if args.stage in ["all", "sculpting"]:
        run_cmd([py, os.path.join(BASE_DIR, "03_inference_and_sculpting", "plot_mgg_sculpting.py"), "--bdt_cut", str(args.bdt_cut)])
        run_cmd([py, os.path.join(BASE_DIR, "03_inference_and_sculpting", "plot_mgg_sculpting_no_dy.py"), "--bdt_cut", str(args.bdt_cut)])
        run_cmd([py, os.path.join(BASE_DIR, "03_inference_and_sculpting", "plot_feature_importance.py")])

    # 5. Correlations
    if args.stage in ["all", "correlations"]:
        run_cmd([py, os.path.join(BASE_DIR, "03_inference_and_sculpting", "plot_correlations.py")])

    # 6. Feature Engineering
    if args.stage in ["all", "features"]:
        run_cmd([py, os.path.join(BASE_DIR, "04_feature_engineering", "plot_engineered_features.py")])

    # 7. Sensitivity & AMS
    if args.stage in ["all", "ams"]:
        run_cmd([py, os.path.join(BASE_DIR, "05_sensitivity_and_ams", "plot_roc_per_mass.py")])
        run_cmd([py, os.path.join(BASE_DIR, "05_sensitivity_and_ams", "calculate_ams_scan.py")])

    print("\n" + "=" * 78)
    print("[PIPELINE COMPLETE] All requested analysis stages finished successfully!")
    print("=" * 78)


if __name__ == "__main__":
    main()
