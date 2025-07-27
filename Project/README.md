<img width="1380" height="1008" alt="image" src="https://github.com/user-attachments/assets/0f169192-067a-4a04-b95a-3d19d8404abd" />Dataset Used from kaggle:https://www.kaggle.com/datasets/alessiocorrado99/animals10


 🐾 Animal Image Classifier using Traditional Machine Learning


An end-to-end machine learning pipeline that classifies animals using handcrafted features (HOG + LBP), PCA for dimensionality reduction, and Random Forest for prediction — deployed with a user-friendly Flask web interface.

---

## 📖 Project Overview

This project aims to classify images of animals such as cats, dogs, butterflies, and more using traditional machine learning methods rather than deep learning. The model extracts handcrafted features (HOG and LBP), reduces dimensions using PCA, and classifies images using a Random Forest Classifier.

The final model is served through a **Flask web app**, where users can upload an image and get a prediction instantly.

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🧠 Feature Engineering | Extracts HOG (Histogram of Oriented Gradients) and LBP (Local Binary Pattern) features |
| 🔻 PCA | Dimensionality reduction for faster training and better generalization |
| 🌲 Random Forest Classifier | Trained to classify animal images |
| 🧪 Evaluation Report | Classification report with accuracy, precision, recall, F1-score |
| 🌐 streamlit App | Upload an image and get prediction instantly |
| 💾 Model Persistence | Saves trained models using `joblib` |
| 📦 Modular Code | Clean structure following best practices |

---

## 🛠️ Technology Stack

- **Language**: Python
- **Libraries**: NumPy, OpenCV, scikit-learn, joblib
- **Web Framework**: streamlit
- **Visualization**: Matplotlib, Seaborn (optional for analysis)
- **Deployment**: Localhost / streamlit / Railway (optional)

---

## 📂 Project Structure

```bash
animal-image-classifier/
│
├── static/                   # Stores uploaded images
│
├── templates/                # HTML templates for Flask
│   └── index.html
│
├── dataset/                  # Contains the image dataset
│   ├── cat/
│   ├── dog/
│   └── ...
│
├── trained_models/           # Contains saved .pkl models
│   ├── rf_model.pkl
│   ├── pca_model.pkl
│   └── label_encoder.pkl
│
├── app.py                    # Flask app
├── feature_extraction.py     # Contains HOG + LBP feature logic
├── model_training.py         # Trains RandomForest + PCA
├── utils.py                  # Helper functions (e.g., image pre-processing)
├── requirements.txt
└── README.md
📁 Dataset Structure
Organized in this way:

bash
Copy
Edit
dataset/
├── cat/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
├── dog/
├── butterfly/
├── chicken/
└── ...
Each folder name becomes a class label, and the images inside are used for training/testing.

🚀 Setup and Installation
🔧 Prerequisites
Python 3.7+

pip

virtualenv (optional but recommended)

💻 Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/animal-image-classifier.git
cd animal-image-classifier
📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
⚙️ Execution Workflow
🔸 Step 1: Train the Model
bash
Copy
Edit
python model_training.py
This will:

Load dataset

Extract HOG + LBP features

Encode labels

Apply PCA

Train RandomForest

Save models using joblib

🔸 Step 2: Launch Flask App
bash
Copy
Edit
python app.py
Navigate to http://localhost:5000/ and upload an image to test.

🌐 Deployment
You can deploy using streamlit
Streamlit (if converted to Streamlit)

🤝 Contributing
Contributions, issues and feature requests are welcome!
Feel free to check the issues page.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Nandikonda Sheshank Reddy
GitHub | LinkedIn


## 🗂️ Dataset Structure
Your dataset folder should look like this:
dataset/
├── butterfly/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── cat/
│   ├── img1.jpg
│   └── ...
├── dog/
│   ├── img1.jpg
│   └── ...
├── chicken/
│   └── ...
├── cow/
│   └── ...
├── elephant/
│   └── ...
├── horse/
│   └── ...
├── sheep/
│   └── ...
├── squirrel/
│   └── ...
└── spider/
    └── ...

