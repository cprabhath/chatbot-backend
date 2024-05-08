To initialize the chatbot Please follow below steps:
if chatbot_model.keras files exists in the directory, no need execute STEP 03.

    STEP 01:
        Create env folder in your directry and activate it. to create that folder please run below command:
            python -m venv env
        then activate it by running below command:
            env\Scripts\activate
    STEP 02:
        Install all the required libraries by running below command:
            pip install -r requirements.txt
    STEP 03:
        Create the chatbot model by running below command:
            python main.py
    STEP 04:
        Run the chatbot by running below command:
            python chatbot.py

Now chatbot has been initialized and you can access it by using API endpoint.
    API Endpoint: http://localhost:5000/get
    Method: POST