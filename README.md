<div align="center">
    <img src="./Images/Churn_Guard.webp" alt="Churn_Guard">
</div>

<div align="center">
  <h1><b>ChurnGuard</b></h1>
</div>

# 📕 Table of Contents

- [📕 Table of Contents](#table-of-contents)
- [🎈 Introduction](#introduction)
- [♻ Data Features](#data-features)
- [📝 Overview](#overview)
- [📊 Exploratory Data Analysis](#exploratory-data-analysis)
- [🔧 Installation](#installation)
- [🚀 Usage](#usage)
- [📝 Article](#article)
- [🤝 Contributing](#contributing)
- [🔏 License](#license)
- [📚 References](#references)
- [👤 Author](#author)

## 🎈 Introduction
The ChurnGuard Project is a machine learning initiative aimed at predicting customer churn for an African telecommunications company (Expresso). It empowers providers to proactively retain customers with predictive analytics. The Project aims to develop a machine learning model capable of predicting customer churn. By leveraging historical customer data, the model assists businesses in identifying customers at risk of leaving, enabling proactive retention strategies.

## ♻ Data Features

- **user_id**: Unique identifier for each customer.
- **REGION**: Region where the customer is located.
- **TENURE**: Duration of the customer's subscription.
- **MONTANT**: Amount spent by the customer.
- **FREQUENCE_RECH**: Frequency of recharges by the customer.
- **REVENUE**: Revenue generated from the customer.
- **ARPU_SEGMENT**: Average Revenue Per User segment.
- **FREQUENCE**: Frequency of usage by the customer.
- **DATA_VOLUME**: Volume of data consumed by the customer.
- **ON_NET**: Calls made to the same network.
- **ORANGE**: Calls made to the Orange network.
- **TIGO**: Calls made to the Tigo network.
- **REGULARITY**: Regularity of the customer's activity.
- **FREQ_TOP_PACK**: Frequency of the customer's top package usage.

## 📝 Overview

The project involves training machine learning models to predict customer churn based on various features extracted from telecom customer data. By deploying these models, businesses can implement targeted strategies to retain customers and reduce churn rates.

## 📊 Exploratory Data Analysis

Explore the dataset used for training the machine learning model. View data visualizations and insights gained from the analysis in the [Exploratory Data Analysis](./notebooks.ipynb) notebook.

![Univariate Analysis](./Images/univariate.png)
![Bivariate Analysis](./Images/bivariate.png)
![Multivariate Analysis](./Images/multivariate.png)


## 🔧 Installation

1. Clone this repository to your local machine.
    
```bash
    git clone https://github.com/Elphoxa/ChurnGuard-Project.git
```

2. Navigate into the repository directory:
   
    ```bash
    cd Churn-Prediction-Project
    ```

3. Create a virtual environment
    
    ```bash
        python -m venv env
    ```

4. Activate the virtual environment
    
    ```bash
    source env/bin/activate

    ```
5. Install the required dependencies

    ```bash
        pip install -r requirements.txt
    ```

## 🚀 Usage

1. Navigate to `main.py`
2. Run the FastAPI application:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

3. Access the API endpoints to predict churn and integrate the API into existing systems.
![alt text](./Images/fastapi_docs.jpg)
![alt text](./Images/fast_2.jpg)


**Docker Image:** The Docker image for the Churn Prediction Project is available on [Docker Hub](https://hub.docker.com/repository/docker/elphoxa56/churn_guard).

## 📝 Article

Read the article on this project [Here](https://www.linkedin.com/pulse/unveiling-customer-churn-guard-efosa-dave-omosigho-oiqzf)

## 🤝 Contributing

If you'd like to contribute to this project, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Make your changes.
- Commit your changes (git commit -am 'Add some feature').
- Push to the branch (git push origin feature/your-feature).
- Create a new Pull Request.

## 🔏 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)

## 👤 Author

🤵 **Efosa Dave Omosigho**
- [GitHub Profile](https://github.com/Elphoxa) 🐙

- [LinkedIn Profile](https://www.linkedin.com/in/efosa-omosigho) 💼
