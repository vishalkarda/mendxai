"""
Main entry point for depression voice detection pipeline.

Usage:
    python main.py --stage extract                              # Extract features
    python main.py --stage train --level subject --model all    # Recommended: subject-level, cross-validated
    python main.py --stage train --level file --model decision_tree
    python main.py --stage evaluate                             # Evaluate file-level models
    python main.py --stage all --level subject --model all      # Full pipeline

Note on --level: this dataset has only 52 subjects with ~29 correlated recordings
each. `--level subject` (recommended) aggregates to one row per subject and
cross-validates — see backend/docs/data_research/04_eda_findings_and_statistics.md
and backend/dev/change_9_*.md for why. `--level file` (the original behavior)
trains on individual recordings with a single file-level train/test split, which
risks leaking a subject's recordings across train/test; kept for backward
compatibility, not recommended for this dataset.
"""
import argparse
from pathlib import Path

import pandas as pd

from mendxai import (
    DataLoader,
    FeatureExtractor,
    ModelTrainer,
    ModelEvaluator,
    config
)
from mendxai.ml.data_loader import verify_data_structure
from mendxai.ml.aggregation import build_subject_level_features
from mendxai.ml.trainer import XGBoostModel


def extract_features():
    """Extract features from audio files."""
    print("\n" + "="*70)
    print("STAGE 1: FEATURE EXTRACTION")
    print("="*70)
    
    # Verify data structure
    if not verify_data_structure():
        print("\n❌ Data structure verification failed. Please organize your data correctly.")
        return False
    
    # Load audio file paths
    data_loader = DataLoader()
    audio_paths, labels = data_loader.load_audio_file_paths()
    
    if len(audio_paths) == 0:
        print("\n❌ No audio files found. Please check your data directory.")
        return False
    
    # Extract features
    extractor = FeatureExtractor()
    features_df = extractor.extract_features_batch(audio_paths, labels)
    
    # Save features
    extractor.save_features(features_df, "features.csv")
    
    print("\n✅ Feature extraction complete!")
    return True


def train_models(model_name: str = "all", tune_hyperparams: bool = False):
    """Train models at the FILE level (single train/val/test split).

    Caution: this dataset has ~29 correlated recordings per subject, and this
    split has no subject-grouping awareness — a subject's recordings can end up
    on both sides of the split. Prefer `train_models_subject_level()`
    (--level subject) for this dataset; see
    backend/docs/data_research/04_eda_findings_and_statistics.md.
    """
    print("\n" + "="*70)
    print("STAGE 2: MODEL TRAINING (file level)")
    print("="*70)
    print(
        "⚠️  File-level split has no subject-grouping — a subject's ~29 recordings "
        "can land on both sides of train/test. Consider --level subject instead."
    )

    # Load features
    extractor = FeatureExtractor()
    try:
        features_df = extractor.load_features("features.csv")
    except FileNotFoundError:
        print("\n❌ Features file not found. Please run feature extraction first:")
        print("   python main.py --stage extract")
        return False
    
    # Prepare data
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)
    
    # Train models
    if model_name == "all":
        trainer.train_all_models(X_train, y_train, X_val, y_val, tune_hyperparams)
    elif model_name == "decision_tree":
        trainer.train_decision_tree(X_train, y_train, tune_hyperparams)
    elif model_name == "xgboost":
        trainer.train_xgboost(X_train, y_train, tune_hyperparams)
    elif model_name == "gradient_boosting":
        trainer.train_gradient_boosting(X_train, y_train, tune_hyperparams)
    elif model_name == "neural_net":
        trainer.train_neural_net(X_train, y_train, X_val, y_val)
    else:
        print(f"\n❌ Unknown model: {model_name}")
        print("   Available models: decision_tree, xgboost, gradient_boosting, neural_net, all")
        return False
    
    print("\n✅ Model training complete!")
    return True


def train_models_subject_level(model_name: str = "all", n_splits: int = 5, n_repeats: int = 20):
    """Train and cross-validate models at the SUBJECT level (recommended for
    this dataset). Aggregates the ~29 recordings per subject into one row per
    subject (mean+std per feature), then runs repeated stratified k-fold CV —
    each row already being one independent subject makes plain StratifiedKFold
    leakage-safe here. The CV itself is the evaluation for this path; there's no
    separate held-out test set (52 subjects is too small to spare one), so no
    `--stage evaluate` step is needed afterward. Each requested model is then
    refit on the full 52-subject dataset and saved.
    """
    print("\n" + "="*70)
    print("STAGE 2: MODEL TRAINING (subject level, cross-validated)")
    print("="*70)

    extractor = FeatureExtractor()
    try:
        features_df = extractor.load_features("features.csv")
    except FileNotFoundError:
        print("\n❌ Features file not found. Please run feature extraction first:")
        print("   python main.py --stage extract")
        return False

    subject_df = build_subject_level_features(features_df)
    print(f"\nSubject-level table: {subject_df.shape[0]} subjects x {subject_df.shape[1]} columns")

    trainer = ModelTrainer()
    model_names = ["decision_tree", "gradient_boosting", "neural_net"] if model_name == "all" else [model_name]

    if "xgboost" in model_names:
        if XGBoostModel is None:
            print(
                "\nxgboost not available in this environment (missing libomp) — skipping. "
                "Use --model gradient_boosting as a stand-in; see backend/dev/change_9_*.md."
            )
            model_names = [m for m in model_names if m != "xgboost"]
        else:
            print("\nNote: cross_validate() does not yet special-case xgboost's own imbalance "
                  "handling (unlike decision_tree/gradient_boosting) — results are still valid, "
                  "just unweighted for this model.")

    cv_results = {}
    for name in model_names:
        results_df = trainer.cross_validate(subject_df, model_name=name, n_splits=n_splits, n_repeats=n_repeats)
        cv_results[name] = results_df

        model, _scaler = trainer.fit_final_model(subject_df, model_name=name)
        config.ensure_output_dirs()
        ext = "pth" if name == "neural_net" else "pkl"
        model.save(config.training.models_dir / f"subject_level_{name}.{ext}")

    # Summary comparison
    summary_rows = []
    for name, df in cv_results.items():
        s = df.drop(columns=["fold"]).agg(["mean", "std"])
        row = {"model": name}
        for metric in s.columns:
            row[f"{metric}_mean"] = s.loc["mean", metric]
            row[f"{metric}_std"] = s.loc["std", metric]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("model")

    print("\n" + "="*70)
    print("SUBJECT-LEVEL CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(summary_df.to_string())

    config.ensure_output_dirs()
    summary_path = config.training.logs_dir / "subject_level_cv_comparison.csv"
    summary_df.to_csv(summary_path)
    print(f"\nSaved comparison to: {summary_path}")

    print("\n✅ Subject-level training + cross-validation complete!")
    return True


def evaluate_models():
    """Evaluate FILE-level trained models against the file-level held-out test
    split. Not applicable to --level subject models — those are cross-validated
    directly in train_models_subject_level(), which has no separate held-out set."""
    print("\n" + "="*70)
    print("STAGE 3: MODEL EVALUATION (file level)")
    print("="*70)
    
    # Load features
    extractor = FeatureExtractor()
    try:
        features_df = extractor.load_features("features.csv")
    except FileNotFoundError:
        print("\n❌ Features file not found. Please run feature extraction first.")
        return False
    
    # Prepare data
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)
    
    # Check if models exist
    models_dir = config.training.models_dir
    available_models = []
    
    if (models_dir / "decision_tree.pkl").exists():
        available_models.append("decision_tree")
    if (models_dir / "xgboost.pkl").exists():
        available_models.append("xgboost")
    if (models_dir / "gradient_boosting.pkl").exists():
        available_models.append("gradient_boosting")
    if (models_dir / "neural_net.pth").exists():
        available_models.append("neural_net")
    
    if not available_models:
        print("\n❌ No trained models found. Please train models first:")
        print("   python main.py --stage train --model all")
        return False
    
    print(f"\nFound trained models: {', '.join(available_models)}")
    
    # Evaluate each model
    evaluator = ModelEvaluator()
    results = []
    
    feature_cols = [col for col in features_df.columns if col not in ['label', 'file_path']]
    
    for model_name in available_models:
        print(f"\nLoading {model_name}...")
        
        if model_name == "decision_tree":
            from mendxai.ml.models.decision_tree import DecisionTreeModel
            model = DecisionTreeModel()
            model.load(models_dir / "decision_tree.pkl")
        elif model_name == "xgboost":
            from mendxai.ml.models.xgboost_model import XGBoostModel
            model = XGBoostModel()
            model.load(models_dir / "xgboost.pkl")
        elif model_name == "gradient_boosting":
            from mendxai.ml.models.gradient_boosting_model import GradientBoostingModel
            model = GradientBoostingModel()
            model.load(models_dir / "gradient_boosting.pkl")
        elif model_name == "neural_net":
            from mendxai.ml.models.neural_net import NeuralNetModel
            input_dim = X_test.shape[1]
            model = NeuralNetModel(input_dim=input_dim)
            model.load(models_dir / "neural_net.pth")
        
        # Evaluate
        metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        # Plot confusion matrix
        y_pred = model.predict(X_test)
        evaluator.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Plot ROC curve
        y_pred_proba = model.predict_proba(X_test)
        evaluator.plot_roc_curve(y_test, y_pred_proba, model_name)
        
        # Plot feature importance (for tree-based models)
        if model_name in ["decision_tree", "xgboost"]:
            evaluator.plot_feature_importance(model, feature_cols, model_name)
    
    # Compare models
    evaluator.compare_models(results)
    
    print("\n✅ Model evaluation complete!")
    print(f"\nResults saved in: {config.training.logs_dir}")
    print(f"Plots saved in: {config.training.figures_dir}")
    
    return True


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Depression Voice Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract features:                python main.py --stage extract
  Train all models (recommended):  python main.py --stage train --level subject --model all
  Train specific model:            python main.py --stage train --level subject --model gradient_boosting
  Evaluate file-level models:      python main.py --stage evaluate
  Full pipeline (recommended):     python main.py --stage all --level subject --model all
  Tune hyperparameters (file level only): python main.py --stage train --level file --model all --tune

Note: --level subject (recommended for this 52-subject dataset) has no separate
--stage evaluate step — cross-validation during training is the evaluation.
        """
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["extract", "train", "evaluate", "all"],
        required=True,
        help="Pipeline stage to run"
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["subject", "file"],
        default="subject",
        help="Training unit: 'subject' (recommended — cross-validated, leakage-safe "
             "for this dataset's ~29-recordings-per-subject structure) or 'file' "
             "(original single-split behavior, kept for backward compatibility). "
             "Default: subject"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["decision_tree", "xgboost", "gradient_boosting", "neural_net", "all"],
        default="all",
        help="Model to train (default: all)"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but better results; --level file only)"
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="CV folds for --level subject (default: 5)"
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="CV repeats for --level subject (default: 20 -> 100 fold-evaluations)"
    )

    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("DEPRESSION VOICE DETECTION PIPELINE")
    print("="*70)
    print(f"Stage: {args.stage}")
    if args.stage in ["train", "all"]:
        print(f"Model: {args.model}")
        print(f"Level: {args.level}")
        if args.tune:
            print("Hyperparameter tuning: ENABLED" if args.level == "file" else
                  "Hyperparameter tuning: not supported for --level subject, ignoring --tune")
    print("="*70)

    def _train(model_name, tune):
        if args.level == "subject":
            return train_models_subject_level(model_name, n_splits=args.n_splits, n_repeats=args.n_repeats)
        return train_models(model_name, tune)

    def _evaluate():
        if args.level == "subject":
            print(
                "\n--level subject has no separate evaluate step — cross-validation during "
                "training IS the evaluation (see backend/results/logs/subject_level_cv_comparison.csv)."
            )
            return True
        return evaluate_models()

    # Execute pipeline stages
    success = True

    if args.stage == "extract":
        success = extract_features()

    elif args.stage == "train":
        success = _train(args.model, args.tune)

    elif args.stage == "evaluate":
        success = _evaluate()

    elif args.stage == "all":
        # Full pipeline
        success = extract_features()
        if success:
            success = _train(args.model, args.tune)
        if success:
            success = _evaluate()
    
    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("❌ PIPELINE FAILED")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
