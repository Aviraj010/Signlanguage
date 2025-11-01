# 🖐️ Sign Language Detection using MediaPipe and OpenCV

## Overview
A machine learning project for real-time sign language gesture recognition using **MediaPipe**, **OpenCV**, and **Scikit-learn**.  
The system detects hand landmarks, preprocesses data, trains a model, and predicts gestures live from the webcam feed.

---

## ✨ Features
- Real-time hand landmark detection using **MediaPipe Hands**
- Automatic data saving in `.csv` format for each gesture
- Model training using **Scikit-learn** (e.g., `RandomForestClassifier`)
- Live prediction display using **OpenCV**
- Easy to expand with custom gestures

---

## 📁 Project Structure
signlanguage/
│
├── utils/
│ ├── hand_detector.py # Handles hand detection using MediaPipe
│ └── preprocessing.py # Handles saving and preprocessing landmark data
│
├── data/ # Folder where collected CSV data is stored
│ ├── A.csv
│ ├── B.csv
│ └── ...
│
├── model/
│ └── sign_model.pkl # Trained model file
│
├── collect_data.py # For capturing hand landmark samples
├── train_model.py # For training model on collected data
├── predict_live.py # For live gesture recognition
└── README.md # Project documentation
---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/signlanguage.git
cd signlanguage
2. Create a Virtual Environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

🧾 Example requirements.txt
opencv-python
mediapipe
numpy
scikit-learn
tensorflow

📚 Library Explanations

opencv-python → Used for capturing webcam video and displaying frames.

mediapipe → Detects and tracks hand landmarks in real-time.

numpy → Handles numerical operations on landmark coordinates.

scikit-learn → For model training and classification.

tensorflow → (Optional) For deep learning-based models if you expand this project later.

🚀 Usage
To collect data:
python collect_data.py

To train the model:
python train_model.py

To run live prediction:
python predict_live.py

🧠 How It Works

The webcam captures your hand using OpenCV.

MediaPipe extracts 3D hand landmarks.

These landmarks are normalized and saved as CSV samples.

Scikit-learn model (like Random Forest) is trained on this data.

During live prediction, the trained model classifies gestures in real-time.

💡 Future Enhancements

Add more gestures for broader recognition.

Implement deep learning models using TensorFlow/Keras.

Create a GUI interface for user-friendly interaction.

👨‍💻 Author

Aviraj Chhetri
B.Sc. Computer Science (Hons) | Salesian College, Siliguri
