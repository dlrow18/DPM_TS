# Predicting Unseen Process Behavior Based on Context Information from Compliance Constraints

## Description
Code accompanying the paper Predicting Unseen Process Behavior Based on Context Information from Compliance Constraints

## Getting started

Create a virtual environment
```
python3 -m venv venv
```

Activate your virtual environment
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Create an augmented transition system
```
python3 generate_ts.py -c config_files/synthetic.config
```

Evaluate prediction results
```
python3 evaluate.py
```

Evaluate prediction results with updates
```
python3 evaluate.py --update=True
```

Test prediction online
```
python3 test_prediction.py
```

## License
LGPL-3.0 license
