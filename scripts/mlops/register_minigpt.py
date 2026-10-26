import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

run_id = "c07e7da55ca34822ad52395cbf3a3ccf"

model_uri = f"runs:/{run_id}/minigpt"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="MiniGPT"
)

print(f"Registered model: {registered_model.name}")
print(f"Version: {registered_model.version}")