"""
Main entry point for depression voice detection pipeline.

Usage:
    python main.py --stage extract                    # Extract features
    python main.py --stage train --model all          # Train all models
    python main.py --stage train --model decision_tree
    python main.py --stage evaluate                   # Evaluate models
    python main.py --stage all --model all            # Full pipeline
"""
import argparse
from pathlib import Path

from mendxai import (
    DataLoader,
    FeatureExtractor,
    ModelTrainer,
    ModelEvaluator,
    config
)
from mendxai.ml.data_loader import verify_data_structure


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
    """Train models."""
    print("\n" + "="*70)
    print("STAGE 2: MODEL TRAINING")
    print("="*70)
    
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
    elif model_name == "neural_net":
        trainer.train_neural_net(X_train, y_train, X_val, y_val)
    else:
        print(f"\n❌ Unknown model: {model_name}")
        print("   Available models: decision_tree, xgboost, neural_net, all")
        return False
    
    print("\n✅ Model training complete!")
    return True


def evaluate_models():
    """Evaluate trained models."""
    print("\n" + "="*70)
    print("STAGE 3: MODEL EVALUATION")
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
  Extract features:           python main.py --stage extract
  Train all models:           python main.py --stage train --model all
  Train specific model:       python main.py --stage train --model xgboost
  Evaluate models:            python main.py --stage evaluate
  Full pipeline:              python main.py --stage all --model all
  Tune hyperparameters:       python main.py --stage train --model all --tune
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
        "--model",
        type=str,
        choices=["decision_tree", "xgboost", "neural_net", "all"],
        default="all",
        help="Model to train (default: all)"
    )
    
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but better results)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("DEPRESSION VOICE DETECTION PIPELINE")
    print("="*70)
    print(f"Stage: {args.stage}")
    if args.stage in ["train", "all"]:
        print(f"Model: {args.model}")
        if args.tune:
            print("Hyperparameter tuning: ENABLED")
    print("="*70)
    
    # Execute pipeline stages
    success = True
    
    if args.stage == "extract":
        success = extract_features()
    
    elif args.stage == "train":
        success = train_models(args.model, args.tune)
    
    elif args.stage == "evaluate":
        success = evaluate_models()
    
    elif args.stage == "all":
        # Full pipeline
        success = extract_features()
        if success:
            success = train_models(args.model, args.tune)
        if success:
            success = evaluate_models()
    
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
