# Obesity Level Prediction - Machine Learning End-to-End Pipeline

This project is an end-to-end Machine Learning web application that predicts a person's obesity risk level based on their eating habits, physical condition, and lifestyle choices. 

It demonstrates a complete AI Engineering workflow: from data preprocessing and handling class imbalance, to model training, evaluation, and deploying the model via a REST/UI interface using Flask.

##  Features
* **Machine Learning Model:** Uses **AdaBoost Classifier** (with Decision Tree as the base estimator) for high-accuracy predictions.
* **Data Balancing:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.
* **Feature Engineering:** Creates composite features (like `Activity_Level`) to improve model performance.
* **Web Interface:** Built with **Flask** to allow users to easily input their health data and get real-time predictions.
* **Prediction Categories:** Predicts 4 classes: Underweight, Normal, Overweight, and Obesity.

##  Tech Stack
* **Language:** Python 3
* **Machine Learning:** `scikit-learn`, `imblearn`
* **Data Manipulation:** `pandas`, `numpy`
* **Web Framework:** `Flask`
* **Model Serialization:** `pickle`

##  Project Structure
* `train.py`: Script for loading data, preprocessing, training the AdaBoost model, and saving the model & scaler as a `.pkl` file.
* `app.py`: The Flask web server that serves the UI and handles the prediction logic.
* `obesity_adaboost_full.pkl`: The trained model and data scaler saved for production use.
* `Obesity_DataSet.xlsx`: The dataset used for training.
* `templates/`: HTML templates for the web interface.
* `static/`: Static assets (CSS/JS) for the frontend.

##  How to Run Locally

1. **Install Dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Re-train the Model:**
   If you want to train the model from scratch, run:
   ```bash
   python train.py
   ```
   *This will generate/overwrite `obesity_adaboost_full.pkl`.*

3. **Start the Web Application:**
   Run the Flask server:
   ```bash
   python app.py
   ```

4. **Access the App:**
   Open your browser and navigate to: `http://localhost:5001`
