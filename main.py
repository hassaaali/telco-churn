import traceback
import pandas as pd
import os
from app.churn_rate import ChurnRatePredictor
from definitions import root_dir
from app.features import feats, target
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def run(random_sample:list):
    """Here is a sample program to predict the churn rate based on the underlying data:
    Note: The following models have to be fine tuned and as of this moment are not fine tuned. 
    Average accuracy noted is 78%.
    """
    data_set = pd.read_csv(os.path.join(root_dir,
                                        "data",
                                        "data_telco_customer_churn.csv"),
                        low_memory=False)
    #random_sample = data_set.iloc[0].iloc[:-1]
    random_sample = pd.DataFrame(random_sample).iloc[0]
    # Kindly refer to the /app/churnrate.py module for operations.
    c = ChurnRatePredictor(data_set=data_set,
                        features=feats,
                        target=target)
    # Pre-Processing the non-integer data and encoding it with mappings.
    c.pre_process()
    # Train test split. Kindly refer to the params in the args.
    c.samples_split()
    # Scaling the data if necessary.
    c.scale()
    # Training the Random forest model. Right now only the most basic version is used for demonstration purposes.
    c.train_random_forest(n_estimators=100,
                        random_state=42,
                        criterion='gini',
                        min_samples_split=2)
    # Evaluating the random forest. You can find the confusion matrix
    # and the classification report in the class's variables.
    c.evaluate_random_forest()
    # In order to resuse the model.
    c.save_random_forest()
    c.load_random_forest()
    # Prediction on a random sample with all data instead of churn. It should return an array with one value.
    print(c.predict_random_forest(new_data=random_sample))
    # Sample run of a very basic neural network. Keep the epochs low since the convergence is early.
    c.train_neural_network(epochs=5, batch_size=1000)
    c.evaluate_neural_network()
    # Saving the NN as a keras file for reuse
    c.save_neural_network()
    c.load_neural_network_model()
    # Predicting on the same data point.
    print(c.predict_neural_network_model(new_data=random_sample))
    return c.predict_neural_network_model(new_data=random_sample)

app = FastAPI(
    title="telco churn predictor",
    description="API to get the inputs required to calculate the teclo churn of a customer",
    version="1.0.0",
    docs_url="/telco-churn-api/docs",
)

class ApplicationData(BaseModel):
    dependents: str
    online_security: str
    online_backup: str
    internet_service: str
    tech_support: str
    contract: str
    paperless_billing: str
    tenure: int
    monthly_charges: float
    device_protection: str

@app.post("/telco-churn-api/get-result", name="Send a sample to ML model", status_code=200)
@app.get("/telco-churn-api/get-result", name="Get prediction result", status_code=200)
def get_result(payload: ApplicationData):
    try:
        data = [{
            "Dependents": payload.dependents,
            "tenure": payload.tenure,
            "OnlineSecurity": payload.online_security,
            "OnlineBackup": payload.online_backup,
            "InternetService": payload.internet_service,
            "DeviceProtection": payload.device_protection,
            "TechSupport": payload.tech_support,
            "Contract": payload.contract,
            "PaperlessBilling": payload.paperless_billing,
            "MonthlyCharges": payload.monthly_charges
        }]
        output = run(data)
        return{'success': f"Will the customer unsubscribe? {output}"}
    except:
        raise HTTPException(status_code=500, detail={'error': f"Error {traceback.format_exc()}"})

@app.get("/")
async def root():
    return {"Uvicorn": "I'm alive"}