# 🚶‍♂️ Human Activity Recognition (HAR) API & Deep Learning System

A production-grade Deep Learning API built with **FastAPI** and **TensorFlow/Keras** to classify human physical activities using smartphone sensor data. The system supports batch processing, allowing users to upload unlabelled sensor datasets in CSV format and instantly download fully annotated prediction reports.

---

## 🚀 Tech Stack
* **Framework:** FastAPI, Uvicorn
* **Deep Learning:** TensorFlow / Keras (Sequential Neural Network)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Serialization:** Joblib (Label Encoder)

---

## 📂 Project Structure
```text
├── main.py              # FastAPI application and prediction pipeline
├── har_model.keras      # Trained Deep Learning classification model
├── encoder.pkl          # Scikit-Learn LabelEncoder for activity mapping
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation


---

 ⚙️ Installation & Setup (Local Running)
Clone the Repository:

Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
Install Dependencies:
Make sure you have Python installed, then install the required libraries:

Bash
pip install -r requirements.txt
Run the FastAPI Server:

Bash
uvicorn main:app --reload
Access the API Documentation:
Open your browser and navigate to:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

🔌 API Endpoints
GET /

Returns a welcome message and system information.

POST /predict

Input: A multipart CSV file containing smartphone sensor features.

Output: A downloadable CSV file containing the original dataset with an appended AI_Prediction column.

---

👤 Author
Pramit Deshwal

B.Tech Computer Science & Engineering (Artificial Intelligence)
