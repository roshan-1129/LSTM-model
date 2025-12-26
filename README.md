# LSTM model
predicts next word using LSTM
# Next Word Prediction using LSTM

This project implements a Next Word Prediction model using Long Short-Term Memory (LSTM) neural networks. The model is trained on Shakespeare's *Hamlet* and deployed as a web application using Streamlit.

## Features

- Predicts the next word based on user input text
- Trained on classic literature (Shakespeare's Hamlet)
- Interactive web interface built with Streamlit
- Uses TensorFlow/Keras for model implementation

## Project Structure

- `app.py`: Streamlit web application for next word prediction
- `experiments.ipynb`: Jupyter notebook containing data preprocessing, model training, and evaluation
- `train.py`: Python script for training the model (alternative to notebook)
- `lstm_text_generator.keras`: Trained LSTM model file
- `tokenizer.pickle`: Saved tokenizer for text preprocessing
- `hamlet.txt`: Raw text data from Shakespeare's Hamlet
- `requirements.txt`: Python dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd lstm-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web App

To start the Streamlit application:

```bash
streamlit run app.py
```

This will launch a web interface where you can enter text and get next word predictions.

### Training the Model (Optional)

If you want to retrain the model:

1. Using the notebook:
   - Open `experiments.ipynb` in Jupyter
   - Run all cells to preprocess data and train the model

2. Using the script:
   ```bash
   python train.py
   ```

## Model Details

- **Architecture**: Sequential LSTM with Embedding layer
- **Training Data**: Shakespeare's Hamlet (preprocessed text)
- **Input**: Sequence of words
- **Output**: Predicted next word
- **Framework**: TensorFlow/Keras

## Dependencies

- streamlit
- tensorflow
- numpy
- nltk
- scikit-learn
- pandas
- matplotlib

## Contributing

Feel free to contribute by:
- Improving the model architecture
- Adding more training data
- Enhancing the web interface
- Optimizing performance

## License

This project is for educational purposes.
