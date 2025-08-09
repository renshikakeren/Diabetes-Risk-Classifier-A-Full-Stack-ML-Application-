This project is an end-to-end machine learning application that predicts the risk of diabetes based on a patient's health parameters. It demonstrates the entire machine learning lifecycle, from data preprocessing and model training to building a web application and containerizing it for deployment.

Key Features & Highlights
Advanced Data Engineering: Performed data cleaning, handled missing values, and engineered a new feature (BMI_Category) to improve model performance.

Multiple Model Comparison: Implemented and compared several machine learning models, including Random Forest, SVM, XGBoost, and Logistic Regression, using a structured approach to find the best-performing classifier.

Hyperparameter Tuning: Utilized GridSearchCV to systematically find the optimal hyperparameters for each model, demonstrating an understanding of model optimization.

Full-Stack Application: Built a simple web interface using Flask to allow users to input health data and receive a real-time prediction from the trained model.

Containerization with Docker: Packaged the entire application, including all dependencies and the model, into a Docker container. This ensures the application is portable and can be deployed consistently on any platform.

Reproducible Environment: Dependencies are managed with a requirements.txt file, ensuring the project can be easily set up and run by others.

Technologies Used
Python: The core programming language for the project.

pandas & numpy: For data manipulation and numerical operations.

Scikit-learn: For machine learning model building and evaluation.

XGBoost: For the Gradient Boosting classifier.

Flask: The web framework used to build the user interface and API.

Docker: For containerizing the application.

HTML & Tailwind CSS: For the front-end design of the web application.

Getting Started
Prerequisites
Python 3.9+

Docker (optional, for containerization)

The Pima Indians Diabetes Dataset (diabetes.csv), which you will need to download and place in the data/ directory.

Installation & Setup
Clone the repository:

git clone https://github.com/your-username/Diabetes-Risk-Classifier-App.git
cd Diabetes-Risk-Classifier-App

Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
pip install -r requirements.txt

Usage
Train the model:

python src/model_training.py

This script will train the models and save the best-performing one to the models/ directory.

Run the web application:

python app.py

Open your browser and navigate to http://localhost:5000 to use the app.

Docker Usage
To run the application using Docker, ensuring a consistent environment:

Build the Docker image:

docker build -t diabetes-classifier-app .

Run the container:

docker run -p 5000:5000 diabetes-classifier-app

The app will be available at http://localhost:5000.
