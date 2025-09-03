# Enhancing Unseen Process Behavior Prediction via Declarative Process Model Mining

## Description
Code accompanying the paper Enhancing Unseen Process Behavior Prediction via Declarative Process Model Mining.  
This project is based on and extends the implementation from [ppm_unseen_constraints](https://github.com/Qian915/ppm_unseen_constraints). 

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

Extract data-aware constraints
```
python DataConstraintsMiner.py
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
