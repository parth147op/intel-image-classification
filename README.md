# 🖼️ Intel Image Classification with CNNs & Transfer Learning  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)

---

## 📌 Project Overview  
This project focuses on building and optimizing **Convolutional Neural Networks (CNNs)** to classify natural scene images from the **Intel Image Classification dataset**.  
The workflow demonstrates:  
- Data preprocessing & augmentation  
- Baseline CNN model  
- Transfer Learning with VGG16  
- Fine-tuning for higher accuracy  
- Hyperparameter tuning with Keras Tuner  
- Model evaluation (confusion matrix, classification report)  
- Explainability with **Grad-CAM/Eigen-CAM**  

---

## 📂 Dataset  
- **Source:** [Intel Image Classification Dataset (Kaggle)](https://www.kaggle.com/puneet6060/intel-image-classification)  
- **Size:** 25,000+ labeled images across 6 categories:  
  - `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`  
- **Splits:**  
  - Train: ~14,000 images  
  - Validation: ~3,000 images  
  - Test: ~3,000 images  

---

## 🛠️ Approach  

### 1. **Data Preparation**
- Verified images for corruption  
- Applied **ImageDataGenerator** with:  
  - Rotation, shifts, zoom, flips, brightness adjustments  
  - Normalization (0–1 scaling)  

### 2. **Baseline CNN**  
- 3 Conv+Pool blocks → Dense layer (256) → Softmax  
- Achieved **80% validation accuracy** in 10 epochs  

### 3. **Transfer Learning (VGG16)**  
- Pretrained on ImageNet  
- Custom dense head trained on Intel dataset  
- Validation accuracy improved to **88.4%**  

### 4. **Fine-Tuning VGG16**  
- Unfrozen last convolutional block (`block5`)  
- Reduced LR (1e-5) + LR scheduling  
- Final **Val Accuracy: 90.4%**, **Test Accuracy: 90.2%**  

### 5. **Hyperparameter Tuning (Keras Tuner)**  
- Tuned number of conv blocks, filters, dense units, dropout, optimizer, and LR  
- Best tuned CNN achieved competitive results  

### 6. **Explainability (Grad-CAM / Eigen-CAM)**  
- Visualized class activation maps  
- Demonstrated **model interpretability**  

---

## 📊 Results  

| Model                       | Validation Accuracy | Test Accuracy |
|---------------------------- |---------------------|---------------|
| Baseline CNN (from scratch) | ~80%                | -             |
| VGG16 (frozen base)         | ~88.4%              | -             |
| VGG16 Fine-tuned            | **90.4%**           | **90.2%**     |

### Classification Report (Fine-Tuned VGG16, Test Set)
```text
              precision    recall  f1-score   support
   buildings      0.91     0.87     0.89       437
      forest      0.97     0.99     0.98       474
     glacier      0.87     0.85     0.86       553
    mountain      0.86     0.84     0.85       525
         sea      0.92     0.92     0.92       510
      street      0.89     0.94     0.92       501

    accuracy                          0.90      3000
```

---

## 📸 Visualizations  

- **Confusion Matrix** → highlights where glacier/mountain misclassifications occur.  
- **Grad-CAM Visualizations** → interpret what CNN “sees” for each prediction.  

> Save plots to `results/` and embed them here, e.g.:
>
> ![Confusion Matrix](results/confusion_matrix.png)  
> ![Grad-CAM Examples](results/gradcam_examples.png)

---

## 📦 Tech Stack  
- **Languages:** Python, NumPy, Pandas  
- **Frameworks:** TensorFlow/Keras  
- **Models:** CNN (custom), VGG16 (transfer learning, fine-tuned)  
- **Visualization:** Matplotlib, Seaborn, Grad-CAM (OpenCV)  
- **Tools:** Jupyter Notebook, Keras Tuner, Streamlit  

---

## 🚀 Deployment Options  

![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Deployable-232F3E?logo=amazonaws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-lightgrey?logo=githubactions&logoColor=white)

- **Dockerized API**: Build with  
  ```bash
  docker build -t intel-cnn-api .
  docker run -p 5000:5000 intel-cnn-api
  ```
- **Cloud Deployment**: AWS/GCP/Azure/Heroku/Render  
- **Production Server**: Use NGINX + Gunicorn instead of dev servers  

---

## 🎨 Frontend Integration  

- **Streamlit/Gradio** demo for drag-and-drop image upload.  
- Optional React/Flask frontend for extended UI.  

---

## 🔄 MLOps Extensions (Future Work)  

- **CI/CD** with GitHub Actions → auto build & deploy  
- **Model Registry** with MLflow / Weights & Biases  
- **Monitoring**: Track performance drift in production  
- **Auto-Rollbacks** if new models underperform  

---

## 🚀 How to Run  

### Notebook
```bash
jupyter notebook notebooks/Intel_Image_Classification.ipynb
```

### Streamlit App
```bash
streamlit run app/app.py
```

---

## 📌 Key Learnings  
- Handling **large image datasets** with pipelines & augmentation  
- Building baseline vs. transfer learning CNNs  
- Fine-tuning pretrained models with **layer freezing & LR scheduling**  
- Performing **hyperparameter optimization** with Keras Tuner  
- Using **explainability tools (Grad-CAM)** to interpret CNNs  

---

## 🏆 Final Note  
This project demonstrates end-to-end expertise in **deep learning for computer vision**, covering everything from preprocessing → modeling → evaluation → interpretability → deployment readiness.  
