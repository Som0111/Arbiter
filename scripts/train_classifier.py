"""Train the sklearn tier classifier on data/classifier_train.json and save data/classifier.pkl."""
import sys

from arbiter.classifier import MODEL_PATH, TRAIN_PATH, cv_report, train_model

if __name__ == "__main__":
    acc = train_model()
    report = cv_report()
    print(f"Training data: {TRAIN_PATH}")
    print(f"5-fold CV accuracy: {acc:.3f}  (model saved to {MODEL_PATH})")
    print(f"{'tier':<10}{'precision':>10}{'recall':>9}{'n':>5}")
    for tier, m in report["per_tier"].items():
        print(f"{tier:<10}{m['precision']:>10.2f}{m['recall']:>9.2f}{m['support']:>5}")
    print(f"{'macro avg':<10}{report['macro_precision']:>10.2f}{report['macro_recall']:>9.2f}")
    sys.exit(0 if acc >= 0.70 else 1)
