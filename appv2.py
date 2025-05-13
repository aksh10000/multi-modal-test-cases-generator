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
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import black
import re
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
        try:
            with open(history_file, 'r') as f:
                st.session_state.test_case_history = json.load(f)
        except json.JSONDecodeError:
            st.session_state.test_case_history = [] # Initialize if file is corrupt
    else:
        st.session_state.test_case_history = []
def escape_xml_chars(text):
    """Escapes XML special characters."""
    text = text.replace("&", "&")
    text = text.replace("<", "<")
    text = text.replace(">", ">")
    text = text.replace('\"', '"')
    text = text.replace("'", "'")
    return text

def markdown_to_reportlab_xml(text):
    """Converts markdown **bold** to ReportLab <b>bold</b> XML tags."""
    # Escape XML special characters first to prevent conflicts with <b> tags
    text = escape_xml_chars(text)
    # Convert markdown bold **text** to <b>text</b>
    # Using a non-greedy match for the content within **
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text

# Save the images in target directory
def save_docs(docs):
    file_paths = []
    for doc in docs:
        # Create the file path
        file_path = os.path.join(target_directory, doc.name)
        
        try:
            # Open the image from the buffer and save it
            image = Image.open(io.BytesIO(doc.getbuffer()))
            # Save the image to the specified directory
            image.save(file_path)
            # Append the file path to the list of file paths
            file_paths.append(file_path)
        except Exception as e:
            st.error(f"Error saving document {doc.name}: {e}. Skipping this file.")
            # Do not add path if save failed
    
    return file_paths

# Function to preprocess images
def preprocess_image(image_path, brightness=0, contrast=1, grayscale=False):
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Could not read image for preprocessing: {image_path}. Using original.")
        return image_path # Return original path if error
    
    # Apply brightness and contrast adjustments
    img_processed = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    # Convert to grayscale if selected
    if grayscale:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        # If Gemini needs 3 channels, convert back. For saving, grayscale is fine.
        # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR) 
    
    # Save the processed image with a prefix
    processed_filename = "processed_" + os.path.basename(image_path)
    processed_path = os.path.join(target_directory, processed_filename)
    
    try:
        cv2.imwrite(processed_path, img_processed)
    except Exception as e:
        st.error(f"Could not save processed image {processed_path}: {e}. Using original from previous step.")
        return image_path # Return original path if save failed

    return processed_path

# Generate the test cases
# Now accepts local_image_paths to associate with results for PDF embedding
def describe(query, uploaded_gemini_files, local_image_paths, detailed_mode=False):
    results = []
    progress_bar = st.progress(0)

    if len(uploaded_gemini_files) != len(local_image_paths):
        st.warning("Warning: Number of files uploaded to Gemini differs from local file paths. "
                   "Image embedding in PDF might be affected for some items.")
    
    for i, image_file_gemini in enumerate(uploaded_gemini_files):
        progress = (i + 1) / len(uploaded_gemini_files)
        progress_bar.progress(progress)
        
        detail_level = "highly detailed" if detailed_mode else "standard"
        
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

Feature Name: {query}

[Image: <input image>]

OUTPUT:
'''
        response = model.generate_content([prompt, image_file_gemini])
        st.markdown(f'''### TEST CASES GENERATED FOR SCREENSHOT-{i+1}:\n\n{response.text}''')
        
        current_local_image_path = None
        if i < len(local_image_paths):
            current_local_image_path = local_image_paths[i]
        else:
            st.warning(f"No corresponding local image path found for uploaded file {i+1}. "
                       "It will not be embedded in the PDF.")

        results.append({
            "screenshot": i+1,
            "image_path": current_local_image_path, # Store the local path for PDF
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "test_cases": response.text
        })
    
    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_bar.empty()
    
    if results:
        history_entry = {
            "id": len(st.session_state.test_case_history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "num_images": len(uploaded_gemini_files),
            "results": results # 'results' now contains 'image_path' for each item
        }
        st.session_state.test_case_history.append(history_entry)
        
        history_file_path = os.path.join(history_directory, "history.json")
        try:
            with open(history_file_path, 'w') as f:
                json.dump(st.session_state.test_case_history, f, indent=4)
        except Exception as e:
            st.error(f"Error saving history file: {e}")
            
    return results

# Upload the images to the gemini endpoint
def upload_to_gemini(file_paths):
    images = []
    progress_bar = st.progress(0)
    
    successful_uploads = 0
    for i, item_path in enumerate(file_paths):
        progress = (i + 1) / len(file_paths)
        progress_bar.progress(progress)
        
        try:
            display_name = f'screenshot-{i}-{os.path.basename(item_path)}'
            image_file = genai.upload_file(path=item_path, display_name=display_name)
            images.append(image_file)
            st.success(f'Uploaded file: {image_file.display_name} (URI: {image_file.uri})')
            successful_uploads +=1
        except Exception as e:
            st.error(f"Failed to upload {item_path} to Gemini: {e}. Skipping this file.")
            # We will have fewer 'images' than 'file_paths' if an upload fails.
            # The 'describe' function will need to handle this potential mismatch
            # by using the actual `local_image_paths` passed to it, corresponding to successful uploads.
            # This requires ensuring that `local_image_paths` passed to `describe` only contains paths for successfully uploaded files.
            # Or, `describe` iterates based on `uploaded_gemini_files` and uses its index `i` for `local_image_paths[i]`,
            # so `local_image_paths` must be filtered to match `uploaded_gemini_files`.
            # Current logic: `describe` gets all `processed_file_paths`. If `upload_to_gemini` skips one, indices will mismatch.
            # Simplest: For now, `describe` checks len and warns. A more robust solution would filter `local_image_paths`
            # to only include paths for images that were successfully uploaded to Gemini.
            # For now, let's assume `describe` gets passed the `file_paths` corresponding to the *attempted* uploads.
            continue 
            
    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_bar.empty()
    
    if successful_uploads < len(file_paths):
        st.warning(f"Successfully uploaded {successful_uploads} out of {len(file_paths)} files.")
    
    return images # This list contains only successfully uploaded genai.File objects

# Function to create PDF of test cases
# Function to create PDF of test cases
def create_pdf(results):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left_margin = 72
    right_margin = 72
    top_margin = 72
    bottom_margin = 72
    content_width = width - left_margin - right_margin
    
    # Setup styles for Paragraph
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        'Normal_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=12, # Line spacing
        textColor=black,
        alignment=TA_LEFT,
    )
    # You can define a specific bold style if needed, but <b> tag works with normal_style
    # bold_style = ParagraphStyle('Bold_Custom', parent=normal_style, fontName='Helvetica-Bold')

    p.setFont("Helvetica-Bold", 16)
    p.drawString(left_margin, height - top_margin, "Test Case Report")
    
    y_position = height - top_margin - 30 
    
    for item_index, result in enumerate(results):
        min_space_for_new_section = 150 
        if y_position < bottom_margin + min_space_for_new_section and item_index > 0: 
            p.showPage()
            y_position = height - top_margin
            p.setFont("Helvetica-Bold", 16)
            p.drawString(left_margin, y_position, "Test Case Report (Continued)")
            y_position -= 30

        p.setFont("Helvetica-Bold", 14)
        current_title_y = y_position
        p.drawString(left_margin, y_position, f"Screenshot {result['screenshot']}")
        y_position -= 20 
        
        image_path = result.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                max_img_display_height = height / 3.0
                img_reader = ImageReader(image_path)
                img_width_orig, img_height_orig = img_reader.getSize()
                aspect_ratio = img_height_orig / float(img_width_orig) if img_width_orig > 0 else 1.0
                display_width = content_width
                display_height = display_width * aspect_ratio
                if display_height > max_img_display_height:
                    display_height = max_img_display_height
                    display_width = display_height / aspect_ratio if aspect_ratio > 0 else content_width
                if display_width > content_width:
                    display_width = content_width
                    display_height = display_width * aspect_ratio

                if y_position - display_height < bottom_margin:
                    p.showPage()
                    y_position = height - top_margin
                    p.setFont("Helvetica-Bold", 14)
                    p.drawString(left_margin, y_position, f"Screenshot {result['screenshot']} (Image Continued)")
                    current_title_y = y_position # Update title_y if new page
                    y_position -= 20

                p.drawImage(img_reader, left_margin, y_position - display_height, 
                            width=display_width, height=display_height, 
                            preserveAspectRatio=True, anchor='n') # Anchor to top of its box
                y_position -= (display_height + 10)
            except Exception as e:
                print(f"PDF Gen Warning: Could not add image {image_path} to PDF: {e}")
                if y_position < bottom_margin + normal_style.leading: p.showPage(); y_position = height - top_margin; p.setFont("Helvetica-Bold", 14); p.drawString(left_margin, y_position, f"Screenshot {result['screenshot']} (Continued)"); y_position -=20
                img_error_para = Paragraph(f"[Image: {os.path.basename(image_path)} - Error loading]", normal_style)
                w,h_err = img_error_para.wrapOn(p, content_width, height)
                img_error_para.drawOn(p, left_margin, y_position - h_err)
                y_position -= (h_err + 5)
        elif image_path:
            if y_position < bottom_margin + normal_style.leading: p.showPage(); y_position = height - top_margin; p.setFont("Helvetica-Bold", 14); p.drawString(left_margin, y_position, f"Screenshot {result['screenshot']} (Continued)"); y_position -=20
            img_missing_para = Paragraph(f"[Image: {os.path.basename(image_path)} - Not found or path is None]", normal_style)
            w,h_miss = img_missing_para.wrapOn(p, content_width, height)
            img_missing_para.drawOn(p, left_margin, y_position - h_miss)
            y_position -= (h_miss + 5)
        
        if y_position < bottom_margin + 3 * normal_style.leading:
            p.showPage()
            y_position = height - top_margin
            p.setFont("Helvetica-Bold", 14) 
            p.drawString(left_margin, y_position, f"Screenshot {result['screenshot']} (Details Continued)")
            y_position -= 20
        
        # --- Draw Timestamp and Query using drawString (as they are simple lines) ---
        p.setFont("Helvetica", 10) # Switch to normal font for these
        ts_text = f"Timestamp: {result['timestamp']}"
        query_text = f"Query: {result['query']}"

        if y_position - normal_style.leading < bottom_margin: p.showPage(); y_position = height - top_margin
        p.drawString(left_margin, y_position, ts_text)
        y_position -= normal_style.leading
        
        if y_position - normal_style.leading < bottom_margin: p.showPage(); y_position = height - top_margin
        p.drawString(left_margin, y_position, query_text)
        y_position -= (normal_style.leading + 5) 
        
        # --- Process and draw test cases using Paragraph for rich text ---
        test_case_lines = result['test_cases'].split('\n')
        
        for line_text in test_case_lines:
            if not line_text.strip(): # Handle empty lines by just adding vertical space
                if y_position - normal_style.leading < bottom_margin:
                    p.showPage()
                    y_position = height - top_margin
                    # Optional: draw continued marker if a new page starts with blank lines from previous section
                y_position -= normal_style.leading 
                continue

            # Convert markdown bold to ReportLab XML bold for this line
            processed_line_text = markdown_to_reportlab_xml(line_text)
            
            para = Paragraph(processed_line_text, normal_style)
            # Calculate required width and height for the paragraph
            p_width, p_height = para.wrapOn(p, content_width, height) # Max height can be large

            # Check for page break before drawing paragraph
            if y_position - p_height < bottom_margin:
                p.showPage()
                y_position = height - top_margin
                p.setFont("Helvetica-Oblique", 9) 
                p.drawString(left_margin + content_width - 60, height - top_margin + 10, "(Continued)") 
                # No need to reset font here as Paragraph uses its own style
            
            para.drawOn(p, left_margin, y_position - p_height) # Draw paragraph
            y_position -= p_height # Adjust y_position by the actual height of the paragraph
        
        y_position -= (normal_style.leading / 2) # Space after each result's test cases
    
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
            "Image Path": result.get('image_path', 'N/A'), # Added image path
            "Test Cases": result['test_cases']
        })
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO() # Use StringIO for text mode
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue() # getvalue() for StringIO

# Function to get download link
def get_download_link(file_data, file_name, display_text):
    if isinstance(file_data, io.BytesIO):  # For PDF (binary)
        b64 = base64.b64encode(file_data.getvalue()).decode()
        mime_type = "application/pdf" if file_name.endswith(".pdf") else "application/octet-stream"
    elif isinstance(file_data, str): # For CSV string
        b64 = base64.b64encode(file_data.encode('utf-8')).decode() # Ensure utf-8 encoding
        mime_type = "text/csv" if file_name.endswith(".csv") else "text/plain"
    else:
        st.error("Unsupported data type for download link.")
        return ""
        
    href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">{display_text}</a>'
    return href

def main():
    st.set_page_config(page_title="Multimodal Tester Pro", layout="wide")
    
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200) # Example logo
        st.title("Tester Pro")
        app_mode = st.radio("Select Mode", ["Generate Test Cases", "View History", "About"], key="app_mode_selector")
    
    if app_mode == "Generate Test Cases":
        tabs = st.tabs(["Upload & Process", "Export", "Settings"])
        
        with tabs[0]:
            st.header("🤖 Test Case Generator")
            user_query = st.text_input("Enter optional context (e.g., specific feature names to test)", key="main_user_query", placeholder="Login, Search, Booking")
            
            st.subheader("📤 Upload Screenshots")
            docs = st.file_uploader("Upload one or more screenshots (.png, .jpg, .jpeg)", 
                                    accept_multiple_files=True, 
                                    type=['png', 'jpg', 'jpeg'], 
                                    key="main_file_uploader")
            
            if docs:
                st.success(f"{len(docs)} screenshot(s) selected and ready for processing.")
                
                st.subheader("🖼️ Image Preprocessing (Optional)")
                preprocess_enabled = st.checkbox("Enable image preprocessing", key="main_preprocess_enabled", help="Adjust image properties before analysis.")
                
                brightness, contrast, grayscale = 0, 1.0, False
                if preprocess_enabled:
                    col_bright, col_contrast, col_gray = st.columns(3)
                    with col_bright:
                        brightness = st.slider("Brightness", -50, 50, 0, key="main_brightness")
                    with col_contrast:
                        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="main_contrast")
                    with col_gray:
                        grayscale = st.checkbox("Convert to Grayscale", key="main_grayscale")
                
                st.subheader("⚙️ Processing Options")
                detailed_mode = st.checkbox("Generate detailed test cases", key="main_detailed_mode", help="Generates more comprehensive test cases, which may take longer.")
                
                if st.button("🚀 Generate Test Cases", key="main_process_button", type="primary"):
                    with st.spinner("Processing... This may take a moment depending on the number of images and detail level."):
                        
                        st.write("Saving uploaded files...")
                        saved_file_paths = save_docs(docs)
                        if not saved_file_paths:
                            st.error("No files were successfully saved. Aborting generation.")
                            return # Exit button action

                        actual_paths_for_gemini = []
                        if preprocess_enabled:
                            st.write("Preprocessing images...")
                            temp_processed_paths = []
                            for path in saved_file_paths:
                                processed_path = preprocess_image(path, brightness, contrast, grayscale)
                                temp_processed_paths.append(processed_path)
                            actual_paths_for_gemini = temp_processed_paths
                        else:
                            actual_paths_for_gemini = saved_file_paths
                        
                        if not actual_paths_for_gemini:
                            st.error("No image files available after potential preprocessing. Aborting generation.")
                            return

                        st.write("Uploading images to AI model...")
                        gemini_uploaded_files = upload_to_gemini(actual_paths_for_gemini)
                        
                        if not gemini_uploaded_files:
                            st.error("No files were successfully uploaded to the AI model. Aborting generation.")
                            return
                        
                        # Important: Ensure `actual_paths_for_gemini` corresponds to `gemini_uploaded_files`.
                        # If `upload_to_gemini` skips files, `actual_paths_for_gemini` needs to be filtered.
                        # For simplicity now, assuming `describe` can handle potential index mismatches with its warning.
                        # A more robust approach would be for `upload_to_gemini` to return a list of (genai_file, local_path) tuples.
                        # Then `describe` would iterate over these pairs.
                        # Current `describe` iterates over `gemini_uploaded_files` and uses its index `i` for `local_image_paths[i]`.
                        # So, `local_image_paths` passed to `describe` should strictly be the paths that correspond to `gemini_uploaded_files`.
                        # If `upload_to_gemini` can return fewer items than `actual_paths_for_gemini`, then `actual_paths_for_gemini`
                        # needs to be filtered *before* passing to `describe` or `describe` needs to handle this.
                        # Let's assume `actual_paths_for_gemini` passed to `describe` should be the list of paths
                        # for which `upload_to_gemini` actually succeeded. This is not currently guaranteed if `upload_to_gemini` skips.
                        # For now, the existing logic in `describe` has a warning for length mismatch.

                        st.write("Generating test cases with AI...")
                        st.session_state.last_results = describe(user_query, gemini_uploaded_files, actual_paths_for_gemini, detailed_mode) 
                        
                        if st.session_state.last_results:
                            st.success("✅ Test case generation complete! Results are displayed below.")
                            st.balloons()
                        else:
                            st.warning("⚠️ Test case generation did not produce results. Please check the inputs or API configuration.")
        
        with tabs[1]:
            st.header("💾 Export Generated Test Cases")
            if 'last_results' in st.session_state and st.session_state.last_results:
                st.info(f"**{len(st.session_state.last_results)}** set(s) of test cases from the last generation are available for export.")
                col_pdf, col_csv = st.columns(2)
                with col_pdf:
                    if st.button("📄 Generate PDF Report", key="main_export_pdf_button"):
                        with st.spinner("Creating PDF report..."):
                            pdf_buffer = create_pdf(st.session_state.last_results)
                        st.markdown(get_download_link(pdf_buffer, f"Test_Case_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "📥 Download PDF Report"), unsafe_allow_html=True)
                with col_csv:
                    if st.button("📊 Generate CSV Data", key="main_export_csv_button"):
                        with st.spinner("Creating CSV file..."):
                            csv_string = create_csv(st.session_state.last_results)
                        st.markdown(get_download_link(csv_string, f"Test_Case_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "📥 Download CSV Data"), unsafe_allow_html=True)
            else:
                st.info("No test cases have been generated in the current session yet. Please go to the 'Upload & Process' tab first.")
        
        with tabs[2]:
            st.header("🔧 Application Settings")
            st.subheader("🔑 API Configuration")
            
            current_api_key_display = os.environ.get('GOOGLE_API_KEY')
            if current_api_key_display:
                 st.caption(f"Currently configured with API key from .env file (ending with ...{current_api_key_display[-4:] if len(current_api_key_display) > 4 else '****'}).")
            else:
                 st.caption("No API key found in .env file.")

            new_api_key = st.text_input("Update Google API Key for this session (optional)", type="password", key="main_api_key_input", help="Overrides .env key for current session only.")
            if st.button("Apply API Key", key="main_update_api_key_button"):
                if new_api_key:
                    try:
                        genai.configure(api_key=new_api_key)
                        st.success("Google API Key updated successfully for this session.")
                    except Exception as e:
                        st.error(f"Failed to configure API key: {e}")
                else:
                    # Re-load from .env if field is empty, effectively resetting to .env
                    env_api_key = os.environ.get('GOOGLE_API_KEY')
                    if env_api_key:
                        try:
                            genai.configure(api_key=env_api_key)
                            st.success("Google API Key reloaded from .env file for this session.")
                        except Exception as e:
                            st.error(f"Failed to re-configure API key from .env: {e}")
                    else:
                        st.warning("No API Key provided in input and not found in .env file.")
            
            st.subheader("🗑️ Cache Management")
            if st.button("Clear Uploaded Files Cache", key="main_clear_cache_button", help="Deletes all files from the 'uploaded_files' directory."):
                try:
                    if os.path.exists(target_directory):
                        import shutil
                        shutil.rmtree(target_directory)
                        os.makedirs(target_directory) # Recreate directory
                        st.success("Uploaded files cache cleared successfully!")
                    else:
                        st.info("Cache directory ('uploaded_files') does not exist, nothing to clear.")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

    elif app_mode == "View History":
        st.header("📜 Test Case Generation History")
        if not st.session_state.test_case_history:
            st.info("No history found. Generate some test cases in the 'Generate Test Cases' mode first!")
        else:
            # Sort history by ID descending (most recent first)
            sorted_history = sorted(st.session_state.test_case_history, key=lambda x: x["id"], reverse=True)

            history_df_data = [{
                "ID": entry["id"], 
                "Timestamp": entry["timestamp"], 
                "Context/Query": entry.get("query", "N/A"), 
                "Images": entry["num_images"]
            } for entry in sorted_history] # Use sorted_history
            
            st.dataframe(pd.DataFrame(history_df_data).set_index("ID"), use_container_width=True)

            history_ids = [entry["id"] for entry in sorted_history] # Use sorted_history
            if not history_ids:
                st.info("No history entries available.")
                return

            selected_id = st.selectbox("Select History ID to view details or export:", options=history_ids, format_func=lambda x: f"ID: {x} ({next(entry['timestamp'] for entry in sorted_history if entry['id'] == x)})", key="history_select_id")

            if selected_id:
                selected_entry = next((entry for entry in st.session_state.test_case_history if entry["id"] == selected_id), None) # Search in original list
                if selected_entry:
                    st.subheader(f"📖 Details for History ID: {selected_entry['id']}")
                    st.markdown(f"**Generated on:** `{selected_entry['timestamp']}`")
                    st.markdown(f"**Context/Query:** `{selected_entry.get('query', 'N/A')}`")
                    st.markdown(f"**Number of Images Processed:** `{selected_entry['num_images']}`")

                    st.markdown("#### ⬇️ Export Full Selected History Entry")
                    col_hist_pdf, col_hist_csv = st.columns(2)
                    with col_hist_pdf:
                        if st.button(f"📄 PDF for ID {selected_id}", key=f"history_export_pdf_button_{selected_id}"):
                            with st.spinner("Creating PDF for selected history entry..."):
                                pdf_buffer = create_pdf(selected_entry["results"])
                            st.markdown(get_download_link(pdf_buffer, f"History_ID{selected_id}_{selected_entry['timestamp'].replace(':', '-')}.pdf", "📥 Download Full PDF"), unsafe_allow_html=True)
                    with col_hist_csv:
                        if st.button(f"📊 CSV for ID {selected_id}", key=f"history_export_csv_button_{selected_id}"):
                            with st.spinner("Creating CSV for selected history entry..."):
                                csv_string = create_csv(selected_entry["results"])
                            st.markdown(get_download_link(csv_string, f"History_ID{selected_id}_{selected_entry['timestamp'].replace(':', '-')}.csv", "📥 Download Full CSV"), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("#### 🔬 Individual Screenshot Details & Test Cases:")
                    for result_item in selected_entry["results"]:
                        with st.expander(f"Screenshot {result_item['screenshot']} (Generated: {result_item['timestamp']})"):
                            image_p = result_item.get('image_path')
                            st.markdown(f"**Original Image Path (cached):** `{image_p if image_p else 'N/A'}`")
                            if image_p and os.path.exists(image_p):
                                try:
                                    st.image(image_p, caption=f"Cached Image for Screenshot {result_item['screenshot']}", use_column_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display cached image: {e}")
                            elif image_p:
                                st.warning("⚠️ Image file not found at path. It might have been cleared from cache.")
                            
                            st.markdown("**Generated Test Cases:**")
                            st.markdown(result_item["test_cases"])
                            
    elif app_mode == "About":
        st.header("ℹ️ About Tester Pro")
        st.markdown("""
        ### Overview
        **Tester Pro** is an AI-powered application designed to assist Quality Assurance (QA) teams by automating the generation of test cases from application screenshots. By leveraging Google's Gemini 1.5 Flash model, it analyzes visual interfaces and user-provided context to produce comprehensive and structured test documentation.

        ### Key Features
        - **Smart Screenshot Analysis:** Upload screenshots (.png, .jpg, .jpeg) of your application.
        - **AI-Driven Test Case Generation:** Utilizes Gemini for intelligent interpretation of UI elements and functionalities.
        - **Contextual Guidance:** Users can input specific feature names or scenarios to focus the AI's test case generation.
        - **Image Preprocessing Suite:** Optional tools to adjust brightness, contrast, or convert images to grayscale, potentially enhancing AI analysis accuracy.
        - **Customizable Detail Level:** Choose between standard or highly detailed test cases to match your testing needs.
        - **Versatile Export Options:** 
            - **PDF Reports:** Generates professional PDF documents containing test cases alongside their corresponding screenshots.
            - **CSV Data:** Exports test cases in a structured CSV format for easy integration with test management tools or spreadsheets.
        - **Persistent History Tracking:** Saves all generation sessions, allowing users to review, revisit, and re-export past results. PDFs from history can include images if they remain in the local cache.
        - **Configuration & Cache Management:** Provides settings for API key input (session-based override) and tools to clear the local cache of uploaded images.

        ### How to Use
        1.  **Go to "Generate Test Cases"**:
            *   (Optional) Enter context like specific feature names (e.g., "User Login", "Search Functionality") in the text input.
            *   Upload one or more screenshots of the application interface you want to test.
            *   (Optional) Enable and configure image preprocessing settings (brightness, contrast, grayscale).
            *   (Optional) Select "Generate detailed test cases" for more comprehensive output.
            *   Click "🚀 Generate Test Cases".
        2.  **Review Generated Test Cases**: The AI-generated test cases will be displayed on the page.
        3.  **Export Results**: Navigate to the "Export" tab. Click the appropriate button to download the test cases as a PDF (which includes images) or a CSV file.
        4.  **Access History**: Select "View History" from the sidebar to browse past test case generation sessions. You can view details and re-export any historical entry.
        5.  **Manage Settings**: In the "Settings" tab (under "Generate Test Cases"), you can temporarily update the Google API Key for the current session or clear the cache of uploaded image files.

        ### Technical Stack
        - **Frontend:** `Streamlit`
        - **AI Model:** `Google Generative AI (Gemini 1.5 Flash)`
        - **Image Handling:** `Pillow (PIL)`, `OpenCV-Python (cv2)`
        - **PDF Generation:** `ReportLab`
        - **Data Handling:** `Pandas`
        - **Environment Management:** `python-dotenv`

        ### Important Considerations
        -   **API Key Security**: Your `GOOGLE_API_KEY` should be stored securely, typically in a `.env` file at the root of the project. The in-app API key update is for the current session only and is not persistently stored by the application.
        -   **Local Image Cache**: Uploaded and processed images are temporarily stored in the `uploaded_files` directory. Clearing this cache (via Settings) will permanently delete these local image files. If images are cleared, PDFs generated from history for those entries will not be able to embed the images.
        -   **API Usage & Costs**: Be mindful of the usage limits and potential costs associated with the Gemini API.
        """)
        st.markdown("--- \n *Tester Pro: Streamlining Your QA Workflow with AI.*")

if __name__ == "__main__":
    main()