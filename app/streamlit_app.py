# This class creates a streamlit app that interacts with the FastAPI service
import requests
import streamlit as st
from PIL import Image

# API URL - Must use the same port (5330) as the FastAPI service
api_url = "http://localhost:5330/predict"

st.title("Satellite Image Classification")

st.write("Upload a satellite image to classify land use and land cover")

# Show available classes
st.subheader("Available Land Cover Classes:")
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

st.markdown("\n".join([f"- {cls}" for cls in classes]))

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                # Get response from the API
                response = requests.post(api_url, files=files)
                # Check the status and file type
                st.write("File type:", uploaded_file.type)
                st.write("Status code:", response.status_code)

                response.raise_for_status()
                # Get the JSON response
                predictions = response.json()["predictions"]

                st.subheader("Classification Results:")
                for pred in predictions:
                    st.write(f"{pred['class']}: {pred['probability']}")

            # Handle errors
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error: {str(e)}")
                st.write("Response content:", response.text)
            except Exception as e:
                st.error(f"Error: {str(e)}")
