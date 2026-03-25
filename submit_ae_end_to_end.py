import os
import subprocess


env = os.environ.copy()
env["AE_PREP_WAIT"] = env.get("AE_PREP_WAIT", "true")

print("[1/2] Submit preprocessing job (SageMaker Processing)")
subprocess.run(["python3", "submit_ae_preprocessing.py"], check=True, env=env)

print("[2/2] Submit training job (SageMaker Training)")
subprocess.run(["python3", "submit_ae_training.py"], check=True, env=env)

print("End-to-end submission completed.")
