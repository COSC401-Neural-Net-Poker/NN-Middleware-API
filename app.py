from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from rlcard.agents import NFSPAgent
from collections import OrderedDict
import torch
import numpy as np
import re


app = Flask(__name__)
CORS(app)

model = NFSPAgent.from_checkpoint(checkpoint=torch.load('cpu_model_new.pt', map_location="cpu"))
model.set_device("cpu")

def preprocess_data(input_data):

    if not isinstance(input_data, str):
        input_data = str(input_data)

    try:
        p = re.compile('(?<!\\\\)\'')
        input_data = p.sub('\"', input_data)
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        # Handle JSON decode error (e.g., logging, returning an error message)
        raise ValueError(f"Invalid JSON input {input_data}")
    
    # Prepare the structure similar to sample_state1
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

    # Populate 'legal_actions'
    if 'legal_actions' in data:
        actions = []
        legal_actions = data['legal_actions']
        for action in legal_actions:
            actions.append((int(action[0]), action[1]))
        legal_actions = OrderedDict(actions)
        preprocessed_data['legal_actions'] = legal_actions
        # preprocessed_data['legal_actions'] = OrderedDict(data['legal_actions'])
        # print(preprocessed_data['legal_actions'])

    # Convert and assign 'obs' to NumPy array
    if 'obs' in data and isinstance(data['obs'], list):
        preprocessed_data['obs'] = np.array(data['obs'], dtype=float)

    # Assuming 'raw_obs' is structured correctly in the incoming data; perform necessary validation
    if 'raw_obs' in data:
        preprocessed_data['raw_obs'] = data['raw_obs']

    # Populate 'raw_legal_actions'
    if 'raw_legal_actions' in data and isinstance(data['raw_legal_actions'], list):
        preprocessed_data['raw_legal_actions'] = data['raw_legal_actions']

    # Populate 'action_record' if available
    if 'action_record' in data and isinstance(data['action_record'], list):
        preprocessed_data['action_record'] = data['action_record']

    return preprocessed_data

@app.route('/model', methods=['POST'])
def run_model():
    data = request.get_data(as_text=True)  # Get JSON payload
    if not data:
        return jsonify({'error': 'No input_string provided'}), 400

    input_string = data

    # Call your Python script with the input string
    data = preprocess_data(input_string)

    action, info = model.eval_step(data)

    return jsonify({'output': action})


@app.route('/health', methods=['GET'])
def health_check():
    return 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

