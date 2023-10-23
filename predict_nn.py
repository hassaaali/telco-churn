import pandas as pd
import os
from app.churn_rate import ChurnRatePredictor
from definitions import root_dir
from app.features import feats, target

"""Here is a sample program to predict the churn rate based on the underlying data:
Note: The following models have to be fine tuned and as of this moment are not fine tuned. 
Average accuracy noted is 78%.
"""
data_set = pd.read_csv(os.path.join(root_dir,
                                    "data",
                                    "data_telco_customer_churn.csv"),
                       low_memory=False)
# Single Sample
random_sample = data_set.iloc[0].iloc[:-1]
# Multiple Samples:
random_samples_multiple = data_set.iloc[:10, :-1]
# Kindly refer to the /app/churnrate.py module for operations.
c = ChurnRatePredictor(data_set=data_set,
                       features=feats,
                       target=target)
# Pre-Processing the non-integer data and encoding it with mappings.
c.pre_process()
# Train test split. Kindly refer to the params in the args.
c.samples_split()
# Sample run of a very basic neural network. Keep the epochs low since the convergence is early.
c.train_neural_network(epochs=100, batch_size=1000)
c.evaluate_neural_network()
# Saving the NN as a keras file for reuse
c.save_neural_network()
c.load_neural_network_model()
# Predicting on the same data point. You can use either of the above random samples to demonstrate.
print(c.predict_neural_network_model(new_data=random_sample))