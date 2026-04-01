from locust import HttpUser, task, constant
import time

class MLInferenceUser(HttpUser):
    wait_time = constant(0.1) 

    @task
    def get_embeddings(self):
        payload = {"text": "В недрах тундры выдры в гетрах тырят в вёдра ядра кедров. Это тестовое предложение для проверки скорости работы нашей языковой модели."}
        
        self.client.post("/embed", json=payload)