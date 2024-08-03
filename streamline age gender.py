{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "afbd987b-06f7-428b-aebf-c748a204b16f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from keras.models import load_model\n",
    "from keras.preprocessing.image import img_to_array\n",
    "from PIL import Image\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0d169350-59e5-4cfc-a10b-9e1ce44e6a04",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n",
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Load the pre-trained models for gender and age\n",
    "gender_model = load_model('gender_classification_model.h5')\n",
    "age_model = load_model('age_regression_model.h5')\n",
    "\n",
    "# Function to preprocess the image\n",
    "def preprocess_image(image):\n",
    "    # Resize image to match model's expected input size\n",
    "    image = image.resize((64, 64))\n",
    "    # Convert image to numpy array\n",
    "    image = img_to_array(image)\n",
    "    # Normalize the image data to [0, 1] range\n",
    "    image = image / 255.0\n",
    "    # Add a fourth dimension for batch size\n",
    "    image = np.expand_dims(image, axis=0)\n",
    "    return image\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "1e605cc0-73a7-4e00-b3b4-f5544975d64c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-08-03 22:51:16.631 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Applications/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Streamlit app\n",
    "st.title(\"Gender and Age Prediction using CNN\")\n",
    "st.write(\"Upload an image, and the model will predict the gender and age.\")\n",
    "\n",
    "# File uploader for image\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Display the uploaded image\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption='Uploaded Image.', use_column_width=True)\n",
    "    st.write(\"\")\n",
    "    st.write(\"Classifying...\")\n",
    "\n",
    "    # Preprocess the image\n",
    "    processed_image = preprocess_image(image)\n",
    "\n",
    "    # Predict gender\n",
    "    gender_prediction = gender_model.predict(processed_image)\n",
    "    gender_class = np.argmax(gender_prediction)\n",
    "    gender_labels = ['Male', 'Female']\n",
    "    gender = gender_labels[gender_class]\n",
    "\n",
    "    # Predict age\n",
    "    age_prediction = age_model.predict(processed_image)\n",
    "    age = int(age_prediction[0][0])\n",
    "\n",
    "    # Show the result\n",
    "    st.write(f\"Prediction: {gender}\")\n",
    "    st.write(f\"Confidence (gender): {gender_prediction[0][gender_class]:.2f}\")\n",
    "    st.write(f\"Predicted Age: {age}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a8c7445-ad9a-4a93-8c23-480b98d9408a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01662cfb-d1e7-4d8b-8caa-6c7ef9d73ed3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
