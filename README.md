# Literacy Classifier for Indian Populations

A web-based application that predicts the literacy status of individuals in India based on demographic features such as age, gender, social group, region, and digital access. This project uses a neural network powered by **TensorFlow** for prediction and features a user-friendly interface built with HTML, CSS, and JavaScript. The backend is implemented using **FastAPI**, providing efficient and scalable API functionality.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Data Source](#data-source)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Performance](#model-performance)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

The Literacy Classifier leverages a trained neural network to predict the likelihood of an individual's literacy status based on demographic features. The goal is to provide an accessible tool for insights into literacy patterns in India, potentially aiding policymakers and researchers in understanding key factors affecting literacy.

The backend API processes user inputs, runs predictions using the trained TensorFlow model, and returns the result in real time. The front end presents a simple, interactive form for users to input demographic details and visualize the results.

---

## Features

- **Demographic-Based Predictions**: Predictions are made based on user-provided demographic features such as age, gender, and region.
- **Neural Network**: Built using TensorFlow, the model is trained on real-world data to ensure accurate predictions.
- **FastAPI Backend**: Provides a scalable, efficient API to serve the predictions.
- **Interactive Frontend**: A clean and simple user interface built with HTML, CSS, and JavaScript.
- **Integration of Machine Learning and Web Development**: Combines AI and web technologies for seamless user interaction.

---

## Data Source

The model was trained on data provided by [Microdata India](https://microdata.gov.in/nada43/index.php/catalog/151/data_dictionary), which includes detailed demographic and literacy-related features. The data was preprocessed and cleaned to ensure high-quality input for the neural network.

---

## Technologies Used

- **Python**: Backend logic and machine learning.
- **TensorFlow**: Neural network implementation.
- **FastAPI**: Backend API framework.
- **HTML, CSS, JavaScript**: Frontend development.
- **Pandas**: Data preprocessing.
- **NumPy**: Numerical operations.
- **GitHub**: Version control and collaboration.

---

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Steps

1. Clone this repository:
    ```bash
    git clone https://github.com/adityanigam14/Literacy-Classifier-for-Indian-Populations.git
    cd Literacy-Classifier-for-Indian-Populations
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn app:app --reload
    ```

5. Open `index.html` in your browser to access the frontend.

---

## Usage

1. Start the server using `uvicorn` as described above.
2. Open the `index.html` file in your web browser.
3. Fill in the demographic details on the form and click "Predict."
4. View the prediction result displayed on the screen.

---

## Model Performance

While the overall accuracy of the model is **86%**, its **recall score for the minority class (Illiteracy)** is an impressive **96%**. This high recall score ensures that the classifier effectively identifies individuals who are likely to be illiterate, which aligns with the primary purpose of this application: to assist in identifying and addressing literacy challenges in vulnerable populations.

The emphasis on recall over accuracy is intentional, as it prioritizes minimizing false negatives—critical when identifying individuals who might otherwise lack access to literacy resources. 

---

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. 

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- The [Microdata India](https://microdata.gov.in/nada43/index.php/catalog/151/data_dictionary) project for providing the dataset.
- TensorFlow and FastAPI communities for excellent tools and documentation.
