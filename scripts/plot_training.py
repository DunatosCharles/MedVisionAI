import json
import matplotlib.pyplot as plt


with open("training_history.json") as f:
    history = json.load(f)


epochs = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.savefig(
    "screenshots/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig(
    "screenshots/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    history["val_f1"],
    label="Validation F1"
)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score")
plt.legend()

plt.savefig(
    "screenshots/f1_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Plots saved!")