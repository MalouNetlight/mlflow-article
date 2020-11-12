from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="address-to-the-server")

# Create a new run
run = client.create_run() # returns mflow.entities.Run

# This is where you would run your experiment 
# For example training a model for machine translation

# Log whatever information you need
client.log_param(run.info.run_id, "author", "Malou Ockenfels")

# Upload your model
client.log_artifact(run.info.run_id, "path/to/model")

# You may want to evaluate your model here
# Once evaluated, you can store the result
client.log_metric(run.info.run_id, "f1-score", 0.99)

# There are plenty of other options, like setting that the model run is over.
client.set_terminated(run.info.run_id)
