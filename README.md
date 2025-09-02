# Enhancing Unseen Process Behavior Prediction via Declarative Process Model Mining

## Description
Code accompanying the paper Enhancing Unseen Process Behavior Prediction via Declarative Process Model Mining

## Getting started

Create a virtual environment
```
python -m venv venv
```

Activate your virtual environment
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Create a DPM transition system
```
python generate_ts.py -c config_files/helpdesk_no_resolve.config
```

Evaluate prediction results
```
python evaluate.py
```

## License
LGPL-3.0 license
