# Import required libraries

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model

model = load_model("best_model.keras")

# Dictionary of traffic sign classes

classes = {
0:'Speed limit (20km/h)',
1:'Speed limit (30km/h)',
2:'Speed limit (50km/h)',
3:'Speed limit (60km/h)',
4:'Speed limit (70km/h)',
5:'Speed limit (80km/h)',
6:'End of speed limit (80km/h)',
7:'Speed limit (100km/h)',
8:'Speed limit (120km/h)',
9:'No passing',
10:'No passing for vehicles over 3.5 tons',
11:'Right-of-way at next intersection',
12:'Priority road',
13:'Yield',
14:'Stop',
15:'No vehicles',
16:'Vehicles over 3.5 tons prohibited',
17:'No entry',
18:'General caution',
19:'Dangerous curve left',
20:'Dangerous curve right',
21:'Double curve',
22:'Bumpy road',
23:'Slippery road',
24:'Road narrows on the right',
25:'Road work',
26:'Traffic signals',
27:'Pedestrians',
28:'Children crossing',
29:'Bicycles crossing',
30:'Beware of ice/snow',
31:'Wild animals crossing',
32:'End of all speed and passing limits',
33:'Turn right ahead',
34:'Turn left ahead',
35:'Ahead only',
36:'Go straight or right',
37:'Go straight or left',
38:'Keep right',
39:'Keep left',
40:'Roundabout mandatory',
41:'End of no passing',
42:'End of no passing by vehicles over 3.5 tons'
}

# Page settings

st.set_page_config(
    page_title="Traffic Sign Classification",
    page_icon="🚦",
    layout="centered"
)

# Sidebar

st.sidebar.title("📌 Project Information")

st.sidebar.write("### Model")
st.sidebar.success("Custom CNN")

st.sidebar.write("### Validation Accuracy")
st.sidebar.success("99.39 %")

st.sidebar.write("### Dataset")
st.sidebar.info("German Traffic Sign Recognition Benchmark (GTSRB)")

st.sidebar.write("### Framework")
st.sidebar.info("TensorFlow + Streamlit")

# Main title

st.title("🚦 Traffic Sign Classification Using CNN")

st.markdown(
"""
Upload a traffic sign image and the model will predict the traffic sign.
"""
)

# Upload image

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and convert RGBA to RGB

    image = Image.open(uploaded_file).convert("RGB")

    # Display image

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict button

    if st.button("Predict"):

        # Resize image

        img = image.resize((30, 30))

        # Convert image to numpy array

        img = np.array(img)

        # Convert RGB to BGR
        # Training images were loaded using OpenCV

        img = img[:, :, ::-1]

        # Normalize pixel values

        img = img / 255.0

        # Add batch dimension

        img = np.expand_dims(img, axis=0)

        # Make prediction

        prediction = model.predict(img)

        # Get top 5 predictions

        top5_indices = np.argsort(prediction[0])[-5:][::-1]

        # Highest probability class

        class_index = top5_indices[0]

        # Confidence score

        confidence = np.max(prediction) * 100

        # Show prediction

        st.success(
            f"🎯 Predicted Sign: {classes[class_index]}"
        )

        # Show confidence

        st.info(
            f"📈 Confidence Score: {confidence:.2f}%"
        )

        # Progress bar

        st.progress(int(confidence))

        # Show top 5 predictions

        st.write("### Top 5 Predictions")

        for i in top5_indices:

            st.write(
                f"{classes[i]} : {prediction[0][i]*100:.2f}%"
            )

# Footer

st.markdown("---")

st.markdown(
"""
### ❤️ Built with TensorFlow and Streamlit

Traffic Sign Classification using Deep Learning (CNN)
"""
)
