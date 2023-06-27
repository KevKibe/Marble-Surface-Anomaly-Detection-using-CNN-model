# Problem Statement
This project addresses a multi-class classification task in computer vision, where the goal is to classify images of marble surfaces into two categories: good or defective.<br> The purpose of this classification is to enable quality control in industries involved in manufacturing marble tops.  

## Training
The model waas trained using a traditional CNN model architecture which had an accuracy of 88% on the validation dataset and a Resnet9 CNN model architecture which had an accuracy of 92% on the validation dataset
## Usage
Upload an image of a marble surface using the provided file uploader in this [Link](https://kevkibe-marble-surface-anomaly-detection-using-cnn-m-app-sesrxd.streamlit.app/)<br>
View the uploaded image and the corresponding prediction of its classification.

# Dataset
The data used to train the model is from [Kaggle](https://www.kaggle.com/datasets/wardaddy24/marble-surface-anomaly-detection-2).
# Motivation
The developed CNN model provides an efficient solution for identifying anomalies in marble surfaces. Its lightweight architecture makes it suitable for deployment on resource-constrained embedded systems. By leveraging this model, manufacturers can automate the quality control process, improving efficiency and reducing manual inspection efforts.
# Further Improvements
The current implementation serves as a solid foundation, but there are opportunities for enhancement:
<li>Increasing the diversity and quantity of the training data can potentially improve the model's performance.
<li>Further optimizations can be explored to adapt the model for specific embedded systems, ensuring efficient inference on devices with limited resources.


  
