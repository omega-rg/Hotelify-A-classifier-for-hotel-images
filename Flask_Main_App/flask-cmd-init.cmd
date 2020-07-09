pip install --upgrade virtualenv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install flask
set FLASK_APP=app.py
git init
flask run