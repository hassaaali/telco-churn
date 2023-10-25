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
    random_sample = pd.DataFrame(random_sample).iloc[0]
    c = ChurnRatePredictor(data_set=data_set,
                        features=feats,
                        target=target)
    c.pre_process()
    c.samples_split()
    c.train_random_forest(n_estimators=100,
                        random_state=42,
                        criterion='gini',
                        min_samples_split=2)
    c.evaluate_random_forest()
    c.save_random_forest()
    c.load_random_forest()
    return c.predict_random_forest(new_data=random_sample)

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

