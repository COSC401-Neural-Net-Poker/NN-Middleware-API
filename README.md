# Poker Prediction Microservice
## Overview
This microservice utilizes a neural network to make poker game predictions. It securely accepts input over HTTPS and processes it to predict outcomes based on its historical self play. Our implementation is running on a google cloud VM behind an NGINX reverse proxy accessible through https://bluffbuddy.lakeviewtechnology.net.

## Features
* Secure HTTPS Interface: Ensures all data transmitted is securely encrypted.
* Neural Network Integration: Utilizes a trained neural network to analyze and predict poker outcomes.
* Real-Time Predictions: Offers predictions in real-time to enhance gameplay speed.

## Prerequisites
* Python 3.11
* Flask
* RLCard
* Pytorch
* NumPy

# Useage
To use the microservice, send a POST request with JSON formatted poker game data:
curl -X POST [https:](https://bluffbuddy.lakeviewtechnology.net) -d DATA -H "Content-Type: application/json"

Data should be formatted as specified by RLCard.
preprocessed_data = {
    'legal_actions': OrderedDict(),
    'obs': None,  # To be converted into a NumPy array
    'raw_obs': {
        'hand': [],
        'public_cards': [],
        'all_chips': [],
        'my_chips': None,
        'legal_actions': [],
        'raise_nums': []
    },
    'raw_legal_actions': [],
    'action_record': []
}

## Model
Model currently in use can be found here:
https://drive.google.com/file/d/1nu5-lvPjaHaEscaV5oqtDv4kCHkP6fXF/view?usp=sharing