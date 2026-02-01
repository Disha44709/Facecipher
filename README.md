Welcome to my project repository! This folder contains a comprehensive record of my journey through Artificial Intelligence, ranging from foundational statistical modeling to advanced Deep Learning and Computer Vision.
├── WEEK 01_Linear_Regression_Scratch/
│   ├── Training_data.xlsx         
│   ├── Test_data.xlsx             
│   └── LR_Implementation.ipynb    
├── WEEK 02_YOLOv8_Object_Detection/
│   ├── Body_Parts_Detection.ipynb
│   ├── data.yaml                  
│   ├── confusion_matrix.png       
│   └── results.png               
└── README.md

WEEK 1: Multiple Linear Regression from Scratch Objective 
The goal was to predict student performance (marks) by analyzing multiple factors like study time, IQ, and socio-economic variables without using high-level ML libraries like Scikit-Learn
Technical ImplementationVectorized Implementation: Used NumPy for matrix multiplications ($y = Xw + b$), ensuring the code is fast and efficient.
Data Preprocessing:Developed a feature_changing function to handle categorical data (Label Encoding).
Implemented Z-Score Normalization to prevent gradient explosion and ensure smooth convergence.
Optimization: Built the Gradient Descent algorithm manually to minimize the Mean Squared Error (MSE) cost function.
Evaluation: Achieved 100% accuracy on the test set within a tolerance of $\pm 0.5$ marks.
Key Concepts CoveredCost Function MinimizationLearning Rate TuningFeature Scaling & Engineering


Week  2: Vehicle & Body Part Detection (YOLOv8)
ObjectiveDeveloping a high-precision Computer Vision model capable of detecting and classifying 16 specific vehicle components and body parts for automated inspection. 
Dataset DetailsSource: Exported via Roboflow.Classes: 16 categories including hood, headlamp, wheel, license_plate, and bumper.Augmentations: Applied Gaussian Blur and Salt & Pepper noise to make the model robust against low-quality camera feeds. 
Implementation WorkflowEnvironment Setup: Configured Ultralytics YOLOv8 in a GPU-accelerated Google Colab environment.
Path Configuration: Optimized data.yaml with absolute paths for seamless data loading.
Model Training: Trained the yolov8n (Nano) architecture for 50+ epochs to achieve optimal Mean Average Precision (mAP).
Performance Analysis: Analyzed the Confusion Matrix to identify any class-wise misclassifications.

Week 3 + project
 Facial Recognition Project

In Week 3, we learned about facial recognition and a new type of neural network called the Siamese Neural Network. Using these concepts, we implemented our final project to identify whether two facial images belong to the same person or different persons.
In this project, a Siamese-based pre-trained face recognition model was used to extract facial features and compare them using a distance metric. The model checks similarity instead of classifying images.
We used Python along with DeepFace, OpenCV, and Matplotlib to load images, process faces, and display results. A base image of the user was compared with another image of the same person having a different orientation, as well as an image of a different person.
The results show that the model successfully matches the same person and correctly identifies a different person.


