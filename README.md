# 🐾 Animal Image Classifier (Cat, Dog, Wild)

A deep learning project built with **PyTorch** and **Streamlit** that classifies images of animals into **Cat 🐱, Dog 🐶, or Wild 🦁**.  
The model is a custom CNN trained on a labeled dataset and deployed using Streamlit for easy web-based predictions.

---
## 🌍 Live Demo
🔗 https://animalclassifier-4akidgmmdiul3ks9yct3zq.streamlit.app/
---

## 🚀 Features
- Upload any image (`.jpg`, `.jpeg`, `.png`)
- Predicts whether the image is **Cat**, **Dog**, or **Wild**
- Simple, interactive web interface built with Streamlit
- Deployed on Streamlit Cloud

---

## 📂 Project Structure
├── app.py # Streamlit app
├── model_state.pth # Saved trained model weights
├── class_names.txt # List of classes (Cat, Dog, Wild)
├── requirements.txt # Project dependencies
└── README.md # Documentation

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Gaurav-S-c/Aniaml_Classifier.git
cd Animal_Classifier
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Model Architecture

**A simple custom CNN with:**

- 3 convolutional layers

- ReLU activations + MaxPooling

- Fully connected layer → output layer with 3 classes

---

## 📷 Usage

- Upload an image of a cat, dog, or wild animal.

- Click Predict.

- See the result .

---

## 🔮 Future Improvements

-**Add more animal classes 🐍🦉🐴**

-**Improve accuracy with transfer learning (ResNet, VGG, etc.)**

-**Deploy as a mobile/web app with FastAPI backend**

---

## 👨‍💻 Author
**Gaurav Sinha**
