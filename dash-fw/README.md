# Franke-Westerhoff ABM Model Explorer

## Getting Started
### Using the demo
This demo lets you interactively explore the Franke-Westerhoff ABM model for strutural
volatility. 

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then
activate it.

```
virtualenv dash-svm-venv

# Windows
dash-svm-venv\Scripts\activate
# Or Linux
source venv/bin/activate
```

Clone the git repo, then install the requirements with pip
```
git clone https://github.com/plotly/dash-sample-apps
cd dash-sample-apps/apps/dash-svm
pip install -r requirements.txt
```

Run the app
```
python app.py
