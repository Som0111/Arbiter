"""Train the sklearn tier classifier and save data/classifier.pkl."""
import sys

from arbiter.classifier import MODEL_PATH, train_model

if __name__ == "__main__":
    acc = train_model()
    print(f"5-fold CV accuracy: {acc:.3f}  (model saved to {MODEL_PATH})")
    sys.exit(0 if acc >= 0.70 else 1)
