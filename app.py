from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import ast
import json

app = Flask(__name__)
CORS(app)

@app.route('/model', methods=['POST'])
def run_model():
    data = request.get_data(as_text=True)  # Get JSON payload
    if not data:
        return jsonify({'error': 'No input_string provided'}), 400

    input_string = data

    # Call your Python script with the input string
    result = subprocess.run(['python', 'make-prediction.py', input_string], capture_output=True, text=True)
    action = result.stdout
    if result.returncode != 0:
        return jsonify({'error': 'Script execution failed', 'details': result.stderr}), 500
    
    output = result.stdout.strip()[-1]
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))

