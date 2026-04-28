import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import io
import platform
import shutil
import os

# Environment-aware Tesseract configuration
if platform.system() == "Windows":
    # Set the path for Windows local environment
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # On Linux (Streamlit Cloud), use shutil to find tesseract in PATH
    tess_cmd = shutil.which('tesseract')
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    else:
        pytesseract.pytesseract.tesseract_cmd = 'tesseract' # Fallback

def preprocess_image(image):
    """Basic image preprocessing to improve OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Thresholding to get binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def perform_ocr(image):
    """Extract text from image using Tesseract."""
    try:
        # Try basic OCR first
        text = pytesseract.image_to_string(image)

        # If results are poor, try preprocessing
        if len(text.strip()) < 5:
            processed = preprocess_image(image)
            text = pytesseract.image_to_string(processed)

        return text
    except Exception as e:
        st.error(f"OCR Engine Error: {e}")
        return None

def main():
    st.set_page_config(page_title="Document OCR Verification", layout="wide")
    st.title("📄 Document OCR & Verification")
    st.markdown("Upload a PAN, Aadhar, or GST document to extract and verify data.")

    # Debug check for Tesseract installation
    import os
    tess_path = pytesseract.pytesseract.tesseract_cmd
    if not os.path.exists(tess_path):
        st.error(f"Tesseract executable not found at: {tess_path}. Please check the path in app.py")
    elif not os.access(tess_path, os.X_OK):
        st.error(f"Tesseract executable found, but Python does not have permission to execute it: {tess_path}")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        # Handle PDF vs Image
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            # For verification, we typically process the first page
            image = images[0]
            st.info("PDF detected. Processing the first page.")
        else:
            image = Image.open(uploaded_file)

        with col1:
            st.subheader("File Preview")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Extracted Data")
            with st.spinner("Applying OCR..."):
                extracted_text = perform_ocr(image)

            if extracted_text:
                st.text_area("Raw OCR Result", extracted_text, height=300)

                st.divider()
                st.subheader("Verification Mode")
                st.markdown("Enter the expected values to compare with the extracted text.")

                # Dynamic verification fields based on document type
                doc_type = st.selectbox("Document Type", ["PAN Card", "Aadhar Card", "GST Certificate"])

                expected_val = st.text_input(f"Enter expected value for {doc_type} verification:")

                if expected_val:
                    if expected_val.lower() in extracted_text.lower():
                        st.success(f"✅ Match Found: '{expected_val}' exists in the document.")
                    else:
                        st.error(f"❌ No Match: '{expected_val}' not found in the extracted text.")
            else:
                st.warning("No text could be extracted from the image.")

if __name__ == "__main__":
    main()
