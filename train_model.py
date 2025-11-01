import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

X, y = [], []

data_dir = "data"

for file in os.listdir(data_dir):
    if file.endswith(".json"):
        label = file.replace(".json", "")
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

            for landmarks in data:
                flattened = []
                for lm in landmarks:
                    # ensure valid landmark structure
                    if isinstance(lm, (list, tuple)) and len(lm) >= 3:
                        flattened.extend(lm[1:3])  # use x and y only
                if len(flattened) == 42:  # 21 landmarks * 2 coords
                    X.append(flattened)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print(" Loaded samples:", X.shape[0])
print(" Features per sample:", X.shape[1])
print(" Classes found:", np.unique(y))

if len(np.unique(y)) < 2:
    print(" Warning: Only one gesture found! Add more gesture samples for training.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f" Model trained successfully! Accuracy: {accuracy*100:.2f}%")

    os.makedirs("model", exist_ok=True)
    with open("model/sign_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Model saved at: model/sign_model.pkl")
