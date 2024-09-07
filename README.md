# multi-modal-test-cases-generator
This repository shows the demonstration of a multi-modal test cases generator system in which we can pass the images as well and optional context and ask the LLM to act as a Tester and produce detailed test cases.

File descriptions:
   1. app.py : Code for streamlit app
   2. requirements.txt : Packages that needs to be installed
   3. ss1 - ss7 : images for testing the apps on red bus app screenshots, each screenshot shows different functionality.

For running this on your machine:
1. Install the requirement.txt
   pip install -r requirements.txt
2. Save your gemini api key with the name "GOOGLE_API_KEY" inside a .env file
3. Run the streamlit app.py
   streamlit run app.py
