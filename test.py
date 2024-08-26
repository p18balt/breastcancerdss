import requests

# Example data matching the required fields
test_data = {
    'Age': 45,
    'BMI': 21.303949,
    'Glucose': 102,
    'Insulin': 13.852,
    'HOMA': 3.485163,
    'Leptin': 7.6476,
    'Adiponectin': 21.056625,
    'Resistin': 23.03408,
    'MCP.1': 552.444
}

response = requests.post('http://localhost:5000/predict', json=test_data)
print(response.json())

