# Import the dependencies
import streamlit as st
from PIL import Image
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import json
from datetime import datetime
import cv2
import numpy as np

# Load the environment variables
load_dotenv()

# Configure gemini model
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel(
    model_name='models/gemini-1.5-flash-latest'
)

# Define directories
target_directory = "uploaded_files"
history_directory = "test_case_history"

# Create directories if they don't exist
for directory in [target_directory, history_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize session state for history tracking
if 'test_case_history' not in st.session_state:
    # Try to load history from file
    history_file = os.path.join(history_directory, "history.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            st.session_state.test_case_history = json.load(f)
    else:
        st.session_state.test_case_history = []

# Save the images in target directory
def save_docs(docs):
    file_paths = []
    for doc in docs:
        # Create the file path
        file_path = os.path.join(target_directory, doc.name)
        # Append the file path to the list of file paths
        file_paths.append(file_path)

        # Open the image from the buffer and save it
        image = Image.open(io.BytesIO(doc.getbuffer()))
        
        # Save the image to the specified directory
        image.save(file_path)
    
    # Return the list of file paths
    return file_paths

# Function to preprocess images
def preprocess_image(image_path, brightness=0, contrast=1, grayscale=False):
    img = cv2.imread(image_path)
    
    # Apply brightness and contrast adjustments
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    # Convert to grayscale if selected
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert back to 3-channel for compatibility
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Save the processed image with a prefix
    processed_path = os.path.join(target_directory, "processed_" + os.path.basename(image_path))
    cv2.imwrite(processed_path, img)
    
    return processed_path

# Generate the test cases
def describe(query, images, detailed_mode=False):
    results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    for i, image_file in enumerate(images):
        # Update progress
        progress = (i + 1) / len(images)
        progress_bar.progress(progress)
        
        # Determine detail level
        detail_level = "highly detailed" if detailed_mode else "standard"
        
        # Build prompt with detail level
        prompt = f'''You are an AI assistant specializing in software quality assurance. Your task is to analyze screenshots of Red Bus app and generate {detail_level} test cases. Each test case should include a description, pre-conditions, testing steps, and expected results.
Instructions:

- Carefully examine each provided screenshot.
- Identify the key functionality or feature shown.
- Create a comprehensive test case following the specified format.
- Ensure your response is clear, detailed, and actionable for QA testers.
- You should create the test cases only for those functions which given as the "Feature Name", if feature name is not provided generate test cases for all the major features available in the screenshot.
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
        # Generate response by passing in the prompt and image file
        response = model.generate_content([prompt, image_file])
        
        # Display the response in the form of markdown
        st.markdown(f'''### TEST CASES GENERATED FOR SCREENSHOT-{i+1}:\n\n{response.text}''')
        
        # Add to results for export
        results.append({
            "screenshot": i+1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "test_cases": response.text
        })
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    time.sleep(0.5)  # Small delay to show completion
    progress_bar.empty()
    
    # Add to history
    if results:
        history_entry = {
            "id": len(st.session_state.test_case_history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "num_images": len(images),
            "results": results
        }
        st.session_state.test_case_history.append(history_entry)
        
        # Save history to file
        with open(os.path.join(history_directory, "history.json"), 'w') as f:
            json.dump(st.session_state.test_case_history, f)
    
    return results

# Upload the images to the gemini endpoint
def upload_to_gemini(file_paths):
    # Save the uploaded images in the images list
    images = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    for i, item in enumerate(file_paths):
        # Update progress
        progress = (i + 1) / len(file_paths)
        progress_bar.progress(progress)
        
        # Upload the images at the gemini uri
        image_file = genai.upload_file(path=item, display_name=f'screenshot-{i}')
        images.append(image_file)
        st.success(f'Uploaded file: {image_file.display_name} as {image_file.uri}')
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    time.sleep(0.5)  # Small delay to show completion
    progress_bar.empty()
    
    # Return the list of uploaded images
    return images

# Function to create PDF of test cases
def create_pdf(results):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Set title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, height - 72, "Test Case Report")
    
    y_position = height - 100
    
    for result in results:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(72, y_position, f"Screenshot {result['screenshot']}")
        y_position -= 20
        
        p.setFont("Helvetica", 10)
        p.drawString(72, y_position, f"Timestamp: {result['timestamp']}")
        y_position -= 15
        
        p.drawString(72, y_position, f"Query: {result['query']}")
        y_position -= 30
        
        # Split test case text into lines and write to PDF
        lines = result['test_cases'].split('\n')
        for line in lines:
            if y_position < 72:  # Check if we need a new page
                p.showPage()
                y_position = height - 72
            
            p.drawString(72, y_position, line)
            y_position -= 15
        
        # Add space between results
        y_position -= 30
        
        # Check if we need a new page for the next result
        if y_position < 150:
            p.showPage()
            y_position = height - 72
    
    p.save()
    buffer.seek(0)
    return buffer

# Function to create CSV of test cases
def create_csv(results):
    data = []
    for result in results:
        data.append({
            "Screenshot": result['screenshot'],
            "Timestamp": result['timestamp'],
            "Query": result['query'],
            "Test Cases": result['test_cases']
        })
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv

# Function to get download link
def get_download_link(file_data, file_name, display_text):
    if isinstance(file_data, io.BytesIO):  # For PDF
        b64 = base64.b64encode(file_data.getvalue()).decode()
    else:  # For CSV string
        b64 = base64.b64encode(file_data.encode()).decode()
    
    href = f'<a href="data:file/{file_name.split(".")[-1]};base64,{b64}" download="{file_name}">{display_text}</a>'
    return href

def main():
    # Set the page config
    st.set_page_config(page_title="Multimodal Tester Pro", layout="wide")
    
    # Sidebar for app navigation
    with st.sidebar:
        st.title("Test Case Generator Pro")
        app_mode = st.radio("Select Mode", ["Generate Test Cases", "View History", "About"])
    
    # Main app logic based on selected mode
    if app_mode == "Generate Test Cases":
        # Create tabs for different functionalities
        tabs = st.tabs(["Upload & Process", "Export", "Settings"])
        
        with tabs[0]:  # Upload & Process tab
            st.header("Test Cases Generator")
            user_query = st.text_input("Enter your optional context here (feature names)")
            
            # For uploading the images
            st.subheader("Upload Documents")
            docs = st.file_uploader("Upload your screenshots here", accept_multiple_files=True)
            
            # Only show processing options if documents are uploaded
            if docs:
                st.success(f"{len(docs)} document(s) uploaded successfully")
                
                # Image preprocessing options
                st.subheader("Image Preprocessing Options")
                preprocess_enabled = st.checkbox("Enable image preprocessing")
                
                if preprocess_enabled:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        brightness = st.slider("Brightness", -50, 50, 0)
                    with col2:
                        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                    with col3:
                        grayscale = st.checkbox("Convert to Grayscale")
                
                # Processing options
                st.subheader("Processing Options")
                detailed_mode = st.checkbox("Generate detailed test cases (takes longer)")
                
                # Process button
                if st.button("Process Images"):
                    with st.spinner("Processing images..."):
                        # Save uploaded docs
                        file_paths = save_docs(docs)
                        
                        # Apply preprocessing if enabled
                        if preprocess_enabled:
                            processed_paths = []
                            for path in file_paths:
                                processed_path = preprocess_image(path, brightness, contrast, grayscale)
                                processed_paths.append(processed_path)
                            file_paths = processed_paths
                        
                        # Upload to Gemini
                        images = upload_to_gemini(file_paths)
                        
                        # Generate test cases
                        st.session_state.last_results = describe(user_query, images, detailed_mode)
                        
                        st.success("Processing complete!")
        
        with tabs[1]:  # Export tab
            st.header("Export Test Cases")
            
            if 'last_results' in st.session_state and st.session_state.last_results:
                # Export options
                st.subheader("Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # PDF Export
                    st.markdown("### Export as PDF")
                    if st.button("Generate PDF"):
                        pdf_buffer = create_pdf(st.session_state.last_results)
                        st.markdown(get_download_link(pdf_buffer, "test_cases.pdf", "Download PDF"), unsafe_allow_html=True)
                
                with col2:
                    # CSV Export
                    st.markdown("### Export as CSV")
                    if st.button("Generate CSV"):
                        csv_string = create_csv(st.session_state.last_results)
                        st.markdown(get_download_link(csv_string, "test_cases.csv", "Download CSV"), unsafe_allow_html=True)
            else:
                st.info("Generate test cases first to enable export options")
        
        with tabs[2]:  # Settings tab
            st.header("Settings")
            
            # API settings
            st.subheader("API Settings")
            api_key = st.text_input("Google API Key (leave empty to use .env file)", type="password")
            
            if api_key and st.button("Update API Key"):
                # Update the API key
                genai.configure(api_key=api_key)
                st.success("API Key updated successfully")
            
            # Clear cache
            st.subheader("Cache Management")
            if st.button("Clear Upload Cache"):
                import shutil
                shutil.rmtree(target_directory)
                os.makedirs(target_directory)
                st.success("Upload cache cleared successfully")
    
    elif app_mode == "View History":
        st.header("Test Case History")
        
        if st.session_state.test_case_history:
            # Create a dataframe for easy display
            history_data = []
            for entry in st.session_state.test_case_history:
                history_data.append({
                    "ID": entry["id"],
                    "Timestamp": entry["timestamp"],
                    "Query": entry["query"],
                    "Images": entry["num_images"]
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df)
            
            # Allow viewing details of a specific entry
            selected_id = st.number_input("Enter ID to view details", min_value=1, max_value=len(st.session_state.test_case_history), step=1)
            
            if st.button("View Details"):
                # Find the selected entry
                selected_entry = next((entry for entry in st.session_state.test_case_history if entry["id"] == selected_id), None)
                
                if selected_entry:
                    st.subheader(f"Details for ID: {selected_id}")
                    st.write(f"Generated on: {selected_entry['timestamp']}")
                    st.write(f"Query: {selected_entry['query']}")
                    st.write(f"Number of images: {selected_entry['num_images']}")
                    
                    # Show test cases for each screenshot
                    for result in selected_entry["results"]:
                        st.markdown(f"### Screenshot {result['screenshot']}")
                        st.markdown(result["test_cases"])
                        
                        # Option to export this specific result
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Export as PDF (Screenshot {result['screenshot']})", key=f"pdf_{selected_id}_{result['screenshot']}"):
                                pdf_buffer = create_pdf([result])
                                st.markdown(get_download_link(pdf_buffer, f"test_case_{selected_id}_{result['screenshot']}.pdf", "Download PDF"), unsafe_allow_html=True)
                        
                        with col2:
                            if st.button(f"Export as CSV (Screenshot {result['screenshot']})", key=f"csv_{selected_id}_{result['screenshot']}"):
                                csv_string = create_csv([result])
                                st.markdown(get_download_link(csv_string, f"test_case_{selected_id}_{result['screenshot']}.csv", "Download CSV"), unsafe_allow_html=True)
        else:
            st.info("No test case history found")
    
    else:  # About mode
        st.header("About Test Case Generator Pro")
        
        st.markdown("""
        ### Overview
        
        This application helps QA teams generate comprehensive test cases by analyzing screenshots of the Red Bus app using Google's Gemini AI.
        
        ### Features
        
        1. **Upload & Process:** Upload screenshots and generate detailed test cases
        2. **Image Preprocessing:** Adjust brightness, contrast, and convert to grayscale
        3. **Export Options:** Save test cases as PDF or CSV
        4. **History Tracking:** Review and export previously generated test cases
        5. **Settings Management:** Configure API keys and manage cache
        
        ### How to Use
        
        1. Upload one or more screenshots of the Red Bus app
        2. Optionally enter specific feature names to focus on
        3. Process the images to generate test cases
        4. Export the results in your preferred format
        
        ### Dependencies
        
        - Google Generative AI (Gemini 1.5 Pro)
        - Streamlit
        - OpenCV for image preprocessing
        - ReportLab for PDF generation
        
        """)
        
        st.markdown("### Contact")
        st.info("For support or feature requests, please contact your system administrator.")

if __name__ == "__main__":
    main()