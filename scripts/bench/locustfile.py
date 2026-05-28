from locust import HttpUser, task, between
import random

class LLMUser(HttpUser):
    wait_time = between(0.1, 1.0)

    #tests
    prompts = [
        "Do you offer discounted energy plans?",
        "Why should I choose your company?",
        "Explain your pricing model",
        "What are your best offers?",
        "Can I save money with your service?"
    ]


    @task
    def chat_completion(self):
        prompt = random.choice(self.prompts)

        payload = {
            "model": "aisales-agent",
            "messages": [
                {"role": "system", "content": "You are a professional sales agent."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

        self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="LLM Chat Request"
        )
