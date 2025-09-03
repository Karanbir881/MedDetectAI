# Introduction
MedDetect AI aims to facilitate real-time disease identification using a deep learning-based web platform for computerized imaging and clinical data analyses.
# Working
The project can be divided into three main stages : train, test, and deploy.
• Training: During this phase, medical image data, for example, X-rays and CT outputs, are gathered from reliable sources like open clinical datasets. They then separate the data into training, validation and test sets. Data Preprocessing techniques: resizing, augmentation is used.
• Testing: The model is tested on the test dataset in order to validate its diagnostic accuracy and robustness, as a final step after training and validation. This stage is the last opportunity to approve the deep learning model, and test its viability to be implemented in the environment in question.
• Deployment: Once trained, the deep learning model is cleaned and deployed onto a cloud server to allow for real-time accessibility via the MedDetect AI platform. The platform enables healthcare professionals and users to upload a medical image, and cloud-based model will analyze it. It offers diagnostic insights, helping detect disease like pneumonia, brain tumors, COVID-19, and breast cancer. Through this deployment, the capabilities of MedDetect AI are being made instantly available to users where they need it most, improving access and enabling decision making in healthcare.
# Clone the repository to your local machine.
Create a conda environment and install the required libraries
conda create -n healthcure python=3.9.13
conda activate healthcure 
pip install opencv-python==4.5.1.48 numpy tensorflow==2.12.0 scikit-learn==0.24.2 imutils==0.5.4 flask==3.0.0 xgboost==2.0.3
When you have successfully created the environment, installed the required libraries, and activated it, simply run the following command in the terminal.
flask run
