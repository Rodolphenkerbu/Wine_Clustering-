Wine Decision Support Analytics Platform
Project Summary
This project is a machine learning decision-support prototype that combines predictive modeling, explainable AI, uncertainty estimation, and API-based model serving. The application uses FastAPI to expose machine learning models through REST APIs, allowing users to obtain predictions, prediction confidence, and model explanations.

Technologies
Python
FastAPI
TensorFlow/Keras
SHAP
LIME
Docker
Pandas
NumPy
Scikit-learn
Interview Notes
Why FastAPI instead of Flask?
FastAPI provides automatic API documentation, built-in validation through Pydantic, type hints, and strong support for machine learning inference services with less boilerplate code than Flask.

What is dependency injection in FastAPI?
Dependency injection allows reusable components such as configuration settings, database connections, and authentication services to be automatically provided to API endpoints instead of being manually created inside each function.

How does SHAP calculate feature contributions?
SHAP uses Shapley Values from game theory to determine how much each feature contributes to a prediction by comparing predictions with and without the feature across multiple feature combinations.

What's the difference between SHAP and LIME?
SHAP provides theoretically consistent feature contributions and can be used globally or locally. LIME explains a specific prediction by creating a local approximation around a single observation.

Why use Monte Carlo Dropout?
Monte Carlo Dropout estimates prediction uncertainty by keeping dropout active during inference and generating multiple predictions. Variation between predictions provides an uncertainty estimate.

How are API endpoints validated?
FastAPI uses Pydantic models to validate incoming requests, ensuring required fields are present and data types are correct before processing.

How does Docker work?
Docker packages the application, dependencies, and runtime environment into a container so the application runs consistently across different systems.

Quick Interview Summary
This project demonstrates the transition from notebook-based machine learning to a deployable decision-support application. It combines predictive modeling, explainable AI, uncertainty estimation, REST APIs, and deployment-oriented design using FastAPI.