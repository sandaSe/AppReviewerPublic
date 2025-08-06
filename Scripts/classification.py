import json
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    input_path = ""

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    y_true = []
    y_pred = []

    skipped = 0

    for idx, item in enumerate(data):
        true_label = item.get("true_label")
        prediction = item.get("prediction", {})
        predicted_label = prediction.get("label")

        print(f"Sample {idx}: true_label={true_label}, predicted_label={predicted_label}")

        if predicted_label is None:
            skipped += 1
            continue

        y_true.append(true_label)
        y_pred.append(predicted_label)

    print(f"Total samples: {len(data)}")
    print(f"Used samples: {len(y_true)} | Skipped due to None predictions: {skipped}")
    print(f"Label distribution in ground truth: {Counter(y_true)}")
    print(f"Label distribution in predictions: {Counter(y_pred)}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\n=== Confusion Matrix ===")
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"Labels: {labels}")
    print(cm)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
