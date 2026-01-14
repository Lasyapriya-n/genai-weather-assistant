**GenAI Weather Assistant**



**Overview:** A GenAi-powered weather assistant that provides real-time weather insights using LLMs and external APIs.



**Features:** 

* Natural language weather queries
* Location-based forecasting
* Modular and dockerized architecture



**Tech Stack:**

* Python
* OpenAI API
* Docker
* REST APIs



**Project Structure:**

|____app.py

|____Dockerfile

|____requirements.txt

|____.env

**How to run:**

```bash

docker build -t genai-weather .

docker run -p 8000:8000 genai-weather

**.env example**
OPEN_API_KEY= your_api_key

WEATHER_API_KEY= your_weather_api_key

