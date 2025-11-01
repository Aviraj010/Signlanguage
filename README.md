Sign Language Detection Using Hand Landmarks

This project is a Sign Language Recognition System built using Python, OpenCV, MediaPipe, and scikit-learn.
It allows users to collect hand landmark data, train a machine learning model, and predict live sign gestures through a webcam feed.

Overview

The system captures 21 hand landmarks using MediaPipe, processes them into numerical features, and trains a classifier to recognize different hand gestures (e.g., letters A–Z or custom signs).
The project consists of three main scripts:

collect_data.py – Collect and save hand landmark data.

train_model.py – Train a machine learning model using saved data.

predict_live.py – Run real-time gesture detection using webcam.

 Features

Real-time hand landmark detection using MediaPipe Hands.

Automatic data saving in .csv format for each gesture.

Model training using scikit-learn (e.g., RandomForestClassifier).

Live prediction display using OpenCV.

Easy to expand with custom gestures.

 Project Structure
signlanguage/
│
├── utils/
│   ├── hand_detector.py        # Handles hand detection using MediaPipe
│   ├── preprocessing.py        # Handles saving and preprocessing landmark data
│
├── data/                       # Folder where collected CSV data is stored
│   ├── A.csv
│   ├── B.csv
│   └── ...
│
├── model/
│   └── sign_model.pkl          # Trained model file
│
├── collect_data.py             # For capturing hand landmark samples
├── train_model.py              # For training model on collected data
├── predict_live.py             # For live gesture recognition
└── README.md                   # Project documentation

 Installation
1. Clone the Repository
git clone https://github.com/yourusername/signlanguage.git
cd signlanguage

2. Create a Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux

3. Install Dependencies
pip install -r requirements.txt

Example requirements.txt
opencv-python
mediapipe
numpy
scikit-learn
tensorflow

 Library Explanations

opencv-python → Used for capturing webcam video and drawing on frames.

mediapipe → Detects and tracks hand landmarks in real time.

numpy → Handles numerical operations and feature vector transformations.

tensorflow → Can be used later for deep learning model integration.

scikit-learn → Trains classical ML models such as RandomForest or SVM.

 Usage
Step 1: Collect Data

Run:

python collect_data.py


Then:

Enter a gesture label (e.g., A)

Press s to save a sample

Press q to quit

Each saved gesture will be stored in data/ as a .csv file.

Step 2: Train Model

Run:

python train_model.py


This script loads all CSV files, trains a classifier, and saves it in model/sign_model.pkl.

Step 3: Run Live Prediction

Run:

python predict_live.py


It will start your webcam and show the predicted sign label in real time.

 Normalization and Preprocessing

During preprocessing:

Landmarks are normalized to ensure consistent scale and position.

Only hand keypoints are stored (x, y, z coordinates).

Feature vectors are flattened and labeled before training.

 Model Details

The model uses RandomForestClassifier for simplicity and accuracy.
Each gesture’s landmark coordinates form a 1D feature vector which is used for classification. 

You can replace it with other models (e.g., SVM, KNN, Neural Network) easily.

 Example Output
Loaded samples: 120
Features per sample: 42
Classes found: ['A', 'B', 'C']
Model trained successfully! Accuracy: 98.5%
Model saved at: model/sign_model.pkl


👨‍💻 Author

Aviraj Chhetri
B.Sc. Computer Science (Hons)
Salesian College, Siliguri
Email:avirajchhetri@gmail.com
