import streamlit as st
from PIL import Image
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel(
        model_name='models/gemini-1.5-pro-latest'
    )


# Step 1: Define the target directory where you want to save the uploaded files
target_directory = "uploaded_files"

# Step 2: Create the directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)
def save_docs(docs):
    file_paths = []
    for doc in docs:
        file_path = os.path.join(target_directory, doc.name)
        file_paths.append(file_path)
        # Display a success message with the file path
        st.success(f"{file_path} ingested")

        # Open the image from the buffer and save it
        image = Image.open(io.BytesIO(doc.getbuffer()))

        # Save the image to the specified directory
        image.save(file_path)
    return file_paths
        
def describe(query:str,images):
    for i,image_file in enumerate(images):
        prompt = f'''You are an AI assistant specializing in software quality assurance. Your task is to analyze screenshots of Red Bus app and generate detailed test cases. Each test case should include a description, pre-conditions, testing steps, and expected results.
Instructions:

- Carefully examine each provided screenshot.
- Identify the key functionality or feature shown.
- Create a comprehensive test case following the specified format.
- Ensure your response is clear, detailed, and actionable for QA testers.
- You should create the test cases only for those functions which given as the "Feature Name", if feature name is not provided generate test cases for all the major features  available in the screenshot.
- If all the features present inside the "Feature Name" are not present in the image then generate the text cases for only those features that are present in the image.
Output Format:
For each feature, provide:

Test Case:

- Description: [Brief overview of what's being tested]
- Pre-conditions: [List any necessary setup or conditions]
- Testing Steps:

[Step 1]
[Step 2]
...
- Expected Result: [What should happen if the feature works correctly]

Examples:

Feature Name: Source and Destination Selection

[Image: A screenshot showing a mobile app interface with fields for entering source and destination cities for a bus journey]

OUTPUT:

Test Case:

- Description: Verify that users can successfully enter and select source and destination cities for their journey.
- Pre-conditions:

The RedBus app is installed and launched
User is on the main search screen


- Testing Steps:

1. Tap on the "From" field
2. Type the first few letters of the source city (e.g., "Mum" for Mumbai)
3. Select "Mumbai" from the autocomplete suggestions
4. Tap on the "To" field
5. Type the first few letters of the destination city (e.g., "Pun" for Pune)
6. Select "Pune" from the autocomplete suggestions


- Expected Result: Both "From" and "To" fields should be populated with the selected cities. The app should be ready for the next step (usually date selection).

Feature Name: Travel Date Selection

[Image: A screenshot showing a calendar interface for selecting a travel date]

OUTPUT:

Test Case:

- Description: Ensure users can select a valid travel date from the calendar interface.
- Pre-conditions:

User has already selected source and destination
The calendar interface is displayed


- Testing Steps:

1. Scroll through the calendar to find the desired month
2. Tap on a date in the future (e.g., 2 weeks from today)
3. Observe the selected date highlight
4. Tap the "Done" or "Confirm" button


- Expected Result: The selected date should be highlighted on the calendar and reflected in the main search interface. The app should proceed to show available buses for the selected route and date.

Feature Name: {query}

[Image: <input image>]

OUTPUT:
'''
    
        response=model.generate_content([prompt, image_file])
        st.markdown(f'''TEST CASES GENERATED FOR FEATURES IN SCREENSHOT-{i}:\n\n
        {response.text}
        ''')

def upload_to_gemini(file_paths):
    images = []
    for i,item in enumerate(file_paths):
        image_file = genai.upload_file(path=item, display_name=f'screenshot-{i}')
        images.append(image_file)
        st.success(f'Uploaded file:{image_file.display_name} as {image_file.uri}')
    return images

def main():
    st.set_page_config(page_title="Multimodal Tester")
    st.header("Test Cases Generator")
    user_query = st.text_input("Enter your optional context here")
    with st.sidebar:
            st.subheader("Your documents")
            docs = st.file_uploader(
                "Upload your documents here and click on 'Process'", accept_multiple_files=True)
            file_paths = save_docs(docs)
            images = upload_to_gemini(file_paths)
            
    if (st.button("Describe Testing Instructions")):
        describe(user_query,images)
if __name__ == "__main__":
    main()
