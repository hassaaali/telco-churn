from flask import Flask, render_template
from flask_session import Session  # https://pythonhosted.org/Flask-Session

app = Flask(__name__)
Session(app)

@app.route('/telco-churn', methods=['GET', 'PUT'])

def telco_churn():
    try:
        return render_template('base.html')
    except Exception as e:
        return render_template('error.html', error=e)

if __name__ == '__main__':
        app.run(host="0.0.0.0", port=8000)