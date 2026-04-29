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
import re

# Environment-aware Tesseract configuration
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_cmd = shutil.which('tesseract')
    pytesseract.pytesseract.tesseract_cmd = tess_cmd if tess_cmd else 'tesseract'

# POppler configuration for Windows
POPPLER_PATH = r'C:\Data\software\poppler\poppler-25.12.0\Library\bin'

def extract_document_id(text, doc_type):
    """Extract and clean document IDs based on known patterns."""
    patterns = {
        "PAN Card": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
        "Aadhar Card": r'[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}',
        "GST Certificate": r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
    }

    pattern = patterns.get(doc_type)
    if not pattern:
        return None

    # Remove common OCR noise (like extra spaces) for Aadhar
    if doc_type == "Aadhar Card":
        text = text.replace(" ", "")
        # Adjust pattern for concatenated Aadhar
        pattern = r'[2-9][0-9]{11}'

    matches = re.findall(pattern, text.upper())
    return matches[0] if matches else None
    
def preprocess_image(image, profile="balanced"):
    """Preprocessing profiles to handle different image qualities."""
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if profile == "conservative":
        # Minimal processing for high-quality images
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    elif profile == "aggressive":
        # Strong processing for very low-res/faded images
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh

    elif profile == "ultra":
        # Ultra-recovery for extremely low-res/pixelated images
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Heavy contrast to force text out of background
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        # Smooth out pixelation
        smoothed = cv2.GaussianBlur(contrast, (3,3), 0)
        # Aggressive threshold
        thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        # Dilate: make text bolder to bridge gaps in broken characters
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        return dilated

    elif profile == "blur_recovery":
        # Specialized for slightly blurry/soft focus images
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 1. Upscale to 2x
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Stronger Sharpening (Laplacian)
        # Instead of just unsharp mask, we use a Laplacian filter to highlight edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        # Combine original gray and sharpened edges
        sharpened = cv2.addWeighted(gray, 0.7, laplacian, 0.3, 0)

        # 3. Contrast boost
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast = clahe.apply(sharpened)

        # 4. Adaptive threshold with tighter C value for blur
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        return thresh

    elif profile == "deep_recovery":
        # Specialized for extremely low-res where standard adaptive fails
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use a mild Gaussian blur to remove high-frequency noise that causes hallucinations
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        # Normalize lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(blurred)
        # Use a more stable thresholding: Otsu's with a slight offset or a fixed threshold
        # Often for low res, a simple binary threshold on a normalized image is safer than adaptive
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    else: # "balanced"
        # Mid-range processing
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        return thresh

def perform_ocr(image):
    """Extract text using a multi-pass strategy with multiple PSM modes for maximum recovery."""
    try:
        best_text = ""
        best_image = None

        # Define the profiles to try
        profiles = ["conservative", "balanced", "aggressive", "ultra", "blur_recovery", "deep_recovery"]
        profiles = ["conservative"]

        # PSM Modes: 3 (Auto), 6 (Uniform block), 11 (Sparse text - best for low res)
        psm_modes = ["--psm 3", "--psm 6", "--psm 11"]
        psm_modes = ["--psm 3"]

        for profile in profiles:
            processed = preprocess_image(image, profile=profile)

            # For each image profile, try different Tesseract segmentation modes
            for psm in psm_modes:
                config = f"{psm}"
                text = pytesseract.image_to_string(processed, config=config)

                # Heuristic: The best result is the one with the most alphanumeric characters
                alnum_count = len([c for c in text if c.isalnum()])
                if alnum_count > len([c for c in best_text if c.isalnum()]):
                    best_text = text
                    best_image = processed

        # ALWAYS store the best processed image for preview, even if text is empty
        # If no text was found at all, we still show the last tried profile's result
        if best_image is None:
            best_image = preprocess_image(image, profile="balanced")

        st.session_state['processed_image'] = best_image
        return best_text

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
        # Reset session state for new upload to avoid "trailing" images
        st.session_state['processed_image'] = None

        # Handle PDF vs Image
        if uploaded_file.type == "application/pdf":
            try:
                # Try using the specified poppler path for Windows
                poppler_path = POPPLER_PATH if platform.system() == "Windows" else None
                images = convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
                # For verification, we typically process the first page
                image = images[0]
                st.info("PDF detected. Processing the first page.")
            except Exception as e:
                st.error(f"PDF Processing Error: {e}")
                st.stop()
        else:
            image = Image.open(uploaded_file)

        # Perform OCR BEFORE defining the UI layout
        # This ensures that the session_state['processed_image'] is updated
        # before the code reaches the rendering block in col1
        with st.spinner("Applying OCR..."):
            extracted_text = perform_ocr(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("File Preview")
            st.image(image, use_container_width=True)

            if 'processed_image' in st.session_state and st.session_state['processed_image'] is not None:
                st.divider()
                st.subheader("OCR Processed View")
                st.image(st.session_state['processed_image'], caption="Preprocessed image for OCR", use_container_width=True)

        with col2:
            st.subheader("Auto-Detection Result")

            # Automatically try to extract the ID from the raw text
            # We try all types to see what matches
            detected_results = []
            for doc_type in ["PAN Card", "Aadhar Card", "GST Certificate"]:
                id_val = extract_document_id(extracted_text, doc_type)
                if id_val:
                    detected_results.append((doc_type, id_val))

            if detected_results:
                for doc_type, id_val in detected_results:
                    st.success(f"✅ **{doc_type}** detected: `{id_val}`")
            else:
                st.warning("No valid Document ID pattern detected. Please verify the raw OCR result.")

            st.divider()
            st.subheader("Extracted Data")
            if extracted_text:
                st.text_area("Raw OCR Result", extracted_text, height=300)

                st.divider()

                st.subheader("Manual Verification")
                st.markdown("If the auto-detection missed it, enter the value manually to verify.")

                doc_type_manual = st.selectbox("Select Document Type", ["PAN Card", "Aadhar Card", "GST Certificate"])
                expected_val = st.text_input(f"Enter expected value for {doc_type_manual} verification:")

                if expected_val:
                    if expected_val.lower() in extracted_text.lower():
                        st.success(f"✅ Match Found: '{expected_val}' exists in the document.")
                    else:
                        st.error(f"❌ No Match: '{expected_val}' not found in the extracted text.")
            else:
                st.warning("No text could be extracted from the image.")

    # Footer for version tracking
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            color: grey;
            font-size: 12px;
            padding: 10px;
        }
        </style>
        <div class="footer">v1.0.5-beta | Document OCR Verification Tool</div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
