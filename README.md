MINI PROJECT 
Bus Buddy: Real-time Crowding Detection and Updates for 
Public Transportation 
AIM: 
The major goal of the Bus Buddy project is to create a system that uses  
real-time human detection to monitor bus occupancy and offer commuters with 
up-to-date, color-coded crowding information so they can make informed travel 
decisions. 
ABSTRACT: 
The Bus Buddy project addresses the common issue of overcrowding in public 
buses, aiming to provide real-time crowding information to commuters. Using 
YOLOv5, a deep learning-based object detection model, the system detects and 
counts passengers inside the bus, classifying the occupancy level as "Low", 
"Moderate", or "High." This information is then communicated to users through 
a mobile or web interface with color-coded indicators (Green, Yellow, Red) that 
reflect the bus's current crowding status. By updating every 30 seconds, the 
system enables passengers to make informed decisions on whether to board the 
next bus or explore alternative travel options, improving the overall commuting 
experience. The project highlights the potential of AI and real-time data to 
enhance public transportation, reduce frustration, and optimize commuter flow. 

CHAPTER 1: INTRODUCTION 
1.1 Problem Statement: 
Everyday commuters are impacted by the ongoing problem of public bus 
overcrowding, which causes annoyance, delays, and an uncomfortable ride. 
When waiting for a bus, passengers frequently worry about whether the next 
one will be packed or if they will have to wait for a less busy one. The 
absence of real-time crowding data in the current public transit networks 
forces commuters to rely on conjecture when making decisions. 
1.2 IMPORTANCE OF BUS BUDDY 
By offering real-time occupancy updates, the Bus Buddy project helps 
mitigate the prevalent problem of crowded public buses, which makes it 
significant. This improves overall travel comfort and cuts down on wait 
times by empowering commuters to make well-informed decisions. The 
system makes public transportation more sustainable and efficient by 
improving passenger safety and bus utilization. 
1.3 OBJECTIVE 
The Bus Buddy project's goal is to use YOLOv5 to detect and count 
passengers in order to offer real-time updates on bus occupancy. Based on the 
number of passengers, the system assigns a crowding level classification of 
Green, Yellow, or Red. This information is updated every 30 seconds. It is 
accessible via a web interface or mobile app, allowing commuters to 
maximize their travel experience, avoid packed buses, and make educated 
decisions. 

CHAPTER 2: LITERATURE REVIEW 
2.1 OVERVIEW 
The Bus Buddy project addresses the problem of crowding in public buses 
by using real-time object detection. Using YOLOv5, a deep learning model 
optimized for passenger detection and counting, the system gives commuters 
precise occupancy data. The number of passengers on a bus is calculated by 
processing real-time photos taken by cameras mounted inside the vehicle 
every 30 seconds. Passengers can then evaluate the bus occupancy before 
boarding by viewing this data through a web interface or mobile app. By 
cutting down on wait times, lowering annoyance, and empowering 
passengers to make better travel choices, Bus Buddy aims to enhance the 
commuter experience. 
2.2 DEEP LEARNING IN REAL-TIME CROWD DETECTION 
In the Bus Buddy project, deep learning is applied through the YOLOv5 
model for real-time crowding detection. The model uses supervised learning 
to identify and count passengers by detecting humans in images captured 
inside the bus. 
• Real-time Human Detection: The YOLOv5 deep learning model is 
used to detect passengers inside the bus. 
• Model Fine-tuning: The model is fine-tuned specifically for bus 
interiors, improving its accuracy in detecting passengers under varying 
conditions. 
• Real-time Processing: The model processes images every 30 seconds 
to update bus crowding levels. 

CHAPTER 3: METHODOLOGY  
3.1 DATA COLLECTION 
In this section, describe how data was gathered, labelled, and prepared for 
the project. Since the project involves detecting crowd in images, this section 
should cover: 
Source 
• The CrowdHuman dataset was chosen for its high-quality annotations of 
pedestrian detection in real-world crowded scenarios. 
Preprocessing: 
1. Dataset Curation: 
o Images containing no people were identified and removed from the 
dataset. 
o The dataset was divided into train and validation sets. 
2. Annotations: 
o Only the "person" class was retained from the dataset annotations. 
o Bounding boxes for individuals were converted from the original 
format to YOLO's required format: (class_id, x_center, y_center, 
width, height), normalized to the image dimensions. 
3. Data Augmentation: 
o Techniques like flipping, scaling, and cropping were applied during 
training to improve model robustness. 
3.2 Model Design and Architecture 
This section provides a detailed explanation of the machine learning model 
architecture and the rationale behind its design. The chosen architecture is based 
on YOLOv5 (You Only Look Once, version 5), a state-of-the-art deep 
learning model tailored for real-time object detection. Each component of the 
architecture has been selected to ensure optimal performance in detecting 
people in crowded environments. 
1. Input Processing 
• The input to the model is an image resized to 640x640 pixels, which 
balances computational efficiency and spatial detail. 
• Images are normalized to ensure numerical stability during training and 
allow the model to generalize better. 
2. Backbone 
The backbone is the feature extraction component of YOLOv5, designed to 
identify important features from the input image. 
• Focus Layer: 
o Extracts patches from the image to enhance spatial density and 
reduce computational load. 
o This helps the model retain high-resolution information critical for 
detecting small objects in crowded settings. 
• CSP-Darknet53: 
o Cross-Stage Partial (CSP) Blocks: Splits and merges feature 
maps, improving gradient flow and reducing computational 
overhead. 
o Convolutional Layers: Stack of layers that extract hierarchical 
features, from edges and textures to more complex patterns like 
crowd structures. 
Rationale: The backbone's lightweight and modular design ensures efficient 
feature extraction while maintaining the ability to detect people across diverse 
crowd densities. 
3. Neck 
The neck aggregates multi-scale features extracted by the backbone to enhance 
detection performance. 
• Path Aggregation Network (PANet): 
o Combines features from different scales to ensure better detection 
of people, especially in crowded scenes with varied object sizes. 
• Feature Pyramid Network (FPN): 
o Enriches feature representations by fusing high-resolution 
(spatially detailed) and low-resolution (semantically rich) features. 
Rationale: This multi-scale feature aggregation enables robust detection of 
individuals in crowds, addressing challenges like overlapping or occluded 
objects. 
4. Detection Head 
The detection head generates predictions for bounding boxes and class 
probabilities. 
• Anchor Boxes: Predefined box templates help the model detect objects of 
varying sizes effectively. 
• Prediction Heads: 
o Operate on three scales to handle objects of different sizes, 
especially critical for detecting small individuals in crowded areas. 
o Outputs include: 
▪ Bounding Box Coordinates: Location of detected 
individuals. 
▪ Objectness Score: Confidence level for the presence of a 
person. 
▪ Class Probability: Likelihood of the detected object 
belonging to the "person" class (ID: 0). 
Rationale: The detection head's multi-scale design ensures accurate detection in 
complex and crowded scenarios. 
5. Dropout Layers 
To mitigate overfitting, dropout layers are integrated during training. These 
layers randomly deactivate a fraction of neurons (e.g., 0.5 dropout rate), forcing 
the model to learn robust patterns rather than relying on specific activations. 
Rationale: Dropout enhances the generalizability of the model, particularly 
when dealing with diverse crowd environments. 
6. Activation Functions 
• Leaky ReLU: Used throughout the model to introduce non-linearity 
while avoiding the vanishing gradient problem. 
• Sigmoid and Softmax: 
o Sigmoid: Converts objectness scores into probabilities. 
o Softmax: Used in the final dense layer to classify the "person" 
class. 
Rationale: These activation functions ensure effective learning of non-linear 
patterns and accurate predictions. 
3.3 Training Process: 
Hyperparameters 
The following settings were used: 
• Image size: 640x640 
• Batch size: 16 
• Learning rate: 0.01 (with cosine learning rate scheduling) 
• Epochs: 50 
• Optimizer: SGD 
Environment 
• Hardware: NVIDIA RTX 3050 
• Software: 
o Python 3.12.1 
o PyTorch 2.4.1 
o YOLOv5 framework (latest version)
 
CHAPTER 4: EXPERIMENTAL SETUP 
4.1 TOOLS AND SOFTWARE 
The Bus Buddy project's software, libraries, and tools for data processing,  
model training, and analysis are described in this part. 
Language for Programming: 
1. Python is the main programming language utilized in this project.  
2. Python was selected because it has robust libraries like TensorFlow, PyTorch, a
 nd OpenCV, and it supports a wide range of machine learning and computer vis
 ion tasks.  
3. Python's enormous community offers a wealth of resources for debugging and 
enhancements, and its readability and simplicity make it perfect for quick proto
 typing and development. 
4.2  Libraries for Machine Learning:  
• YOLOv5 (PyTorch): Used for real-time human detection and passenger 
counting in bus interiors. 
• NumPy: Handles numerical operations and array manipulations for image 
and data processing. 
• OpenCV: Performs image capture, processing, and manipulation for 
preparing data for the model. 
• Matplotlib: Visualizes model performance and training results through 
graphs and plots. 
• Pandas: Manages and processes structured data, particularly for training 
and evaluation datasets. 
4.3  Hardware Specifications: 
1. CPU 
• Processor: Intel Core i5/i7 (or equivalent) 
• Clock Speed: 2.6 GHz or higher 
• Cores and Threads: Quad-core or higher (minimum 8 threads) 
2. GPU 
• Graphics Card: NVIDIA GeForce RTX 3050 
• VRAM: 6 GB 
• CUDA Support: CUDA 11.8 (or compatible) 
3. RAM 
• Capacity: 16 GB 
• Type: DDR4 (or higher) 
4. Storage 
• Type: SSD (Solid-State Drive) 
• Capacity: 512 GB (minimum) 
5. Software Environment 
• Operating System: Windows 10 (64-bit) 
• Python Version: Python 3.12.1 
• Deep Learning Framework: PyTorch 2.4.1 
4.4 Training parameters: 
1. Image Size (--img) 
• Value: 640 
2. Batch Size (--batch) 
• Value: 16 
3. Number of Epochs (--epochs) 
• Value: 50 
4. Learning Rate (lr0) 
• Value: 0.01 
5. Learning Rate Scheduler (lrf) 
• Value: 0.01 
6. Momentum 
• Value: 0.937 
7. Weight Decay (weight_decay) 
• Value: 0.0005 
8. Data Augmentation Parameters 
• Flip Horizontal (fliplr): 0.5 
• Scale (scale): 0.5 
• Translate (translate): 0.1 
9. Dropout Rate 
• Value: 0.5 
10. Optimizer (--optimizer) 
• Value: Stochastic Gradient Descent (SGD) 
4.5 Experimental Procedure 
This section outlines the steps taken to train and evaluate the YOLOv5 
model for crowd detection. 
Dataset Preparation 
• Source: CrowdHuman dataset with a focus on the "person" class. 
• Preprocessing: Images resized to 640x640, annotations converted to 
YOLO format. 
• Splits: 80% training, 10% validation, 10% testing. 
Model Initialization 
• Base Model: YOLOv5s with pretrained weights (yolov5s.pt) from 
COCO. 
• Hyperparameters: Batch size = 16, epochs = 50, learning rate optimized 
using a cosine scheduler. 
Training 
• Environment: Training on an NVIDIA GTX 1650 GPU using PyTorch. 
• Execution: Monitored metrics like mAP, Precision, and Recall during 
training. 
Testing and Evaluation 
• Metrics: Precision, Recall, mAP, and inference time assessed on the test 
set. 
• Post-Processing: Applied confidence threshold (0.25) and Non-Maximum 
Suppression (NMS). 
3.7 Challenges and Limitations 
Challenges 
1. Hardware Constraints: Limited GPU memory (NVIDIA GTX 1650) 
restricted batch size and slowed training. 
2. Dataset Issues: Inconsistent annotations and overlapping bounding boxes 
required preprocessing. 
3. Model Performance: Struggled with occluded individuals and precise 
detection in dense crowds. 
Limitations 
1. Real-Time Inference: Inference speed on lower-end GPUs may not meet 
real-time requirements. 
2. Generalization: Performance drops in different lighting, angles, or crowd 
densities. 
3. Single-Class Detection: Focused only on detecting people, limiting 
broader use cases.

CHAPTER 5: RESULTS AND DISCUSSION 
5.1 MODEL PERFORMANCE EVALUATION 
This section presents the evaluation metrics used to assess the YOLOv5 model's 
performance on the test dataset for crowd detection: 
• Accuracy: The model's accuracy score reflects the proportion of correctly 
predicted bounding boxes and classes. While it provides a general 
performance measure, it may not account for imbalances in detection 
scenarios. 
• Precision, Recall, and F1 Score: These metrics are crucial for evaluating 
object detection: 
o Precision: Measures the proportion of correctly detected "Person" 
instances among all positive predictions. This indicates how 
accurately the model identifies people in crowded scenes. 
o Recall: Evaluates the model's ability to detect all people in the 
dataset, which is vital for safety-critical applications like crowd 
monitoring. 
o F1 Score: Combines precision and recall into a single metric, 
providing a balanced performance measure, especially when 
datasets have varying densities. 
• Confusion Matrix: Visualizes the true positives, false positives, and false 
negatives for the "Person" class, helping identify specific areas for 
improvement, such as missed detections in occluded or low-light 
scenarios. 
5.2 TRAINING AND VALIDATION RESULTS 
This section analyzes the performance trends during training: 
• Training and Validation Loss: Plots of training and validation loss across 
epochs reveal how well the model fits the data. Key observations: 
o Underfitting: High losses indicate the model struggles to learn 
patterns in the data. 
o Overfitting: Divergence between low training loss and high 
validation loss suggests the model memorizes training data but fails 
to generalize. 
• Training and Validation mAP: Mean Average Precision (mAP) measures 
the model's ability to localize and classify bounding boxes accurately. 
Trends in mAP across epochs indicate the model's robustness on both 
training and validation datasets. 
5.3 COMPARATIVE ANALYSIS 
This section compares the YOLOv5-based crowd detection model with other 
methods: 
• Comparison with Existing Approaches: Discuss how the YOLOv5 
approach performs in terms of speed, accuracy, and computational 
efficiency compared to other methods like Faster R-CNN or SSD. 
Highlight YOLOv5's real-time performance advantages. 
• Trade-offs: 
o Detection Speed: YOLOv5 excels in inference time, making it 
suitable for real-time applications. 
o Detection Accuracy: Performance may drop in highly dense or 
occluded scenes compared to methods designed for extreme crowd 
conditions. 
5.4 INTERPRETATION OF RESULTS 
This section interprets the model’s strengths and weaknesses: 
• Effectiveness in Crowd Detection: The YOLOv5 model demonstrates 
high precision and recall for detecting people in moderately crowded 
scenarios but faces challenges with severe occlusions or small-scale 
objects. 
• Limitations: 
o Occlusion Handling: The model struggles to detect partially visible 
individuals in dense crowds. 
o Lighting and Perspective: Performance varies under extreme 
lighting or unusual camera angles. 
5.5 IMPLICATIONS OF FINDINGS 
This section explores potential applications and broader implications of the 
results: 
• Potential Applications: 
o Real-time crowd monitoring systems for events, public 
transportation, and safety management. 
o Integration into apps for tracking bus occupancy, providing users 
with live crowd insights. 
• Practical Considerations: 
o The model’s dependency on high-quality image data may limit its 
application in low-resolution or poorly lit environments. 
o Ethical considerations include ensuring the system respects privacy 
regulations and is used responsibly. 
5.6 SUMMARY OF RESULTS AND DISCUSSION 
This chapter highlighted the YOLOv5 model's strengths, including real-time 
detection capabilities and high precision in moderately crowded scenes. Key 
limitations include challenges with occlusion and extreme crowd densities. 
These insights guide future work, emphasizing optimization for dense and 
diverse real-world scenarios. 

CHAPTER 6: CONCLUSION AND FUTURE WORK: 
6.1 SUMMARY OF FINDINGS 
This section provides a concise summary of the key findings from the study, 
bringing together the results from previous chapters. 
• Overview of Objectives and Approach: The main objective of this project 
was to develop a machine learning model capable of detecting the 
number of people in a crowd using visual data. The approach involved 
data collection from sources like CrowdHuman and COCO datasets, 
preprocessing the data, designing a model using YOLOv5, and training it 
on the annotated images. 
• Key Results: The model's performance was evaluated based on standard 
metrics such as accuracy, precision, recall, and F1 score. The results 
demonstrate that the model is effective at detecting and counting the 
number of people in various crowd scenarios. The model showed strong 
performance in distinguishing between different crowd densities, with 
particular success in more crowded environments. 
• Significance of Results: The model's performance signifies a step forward 
in automated crowd detection, offering potential applications in areas 
such as public safety, event management, and transportation. It is a 
valuable tool for monitoring crowd sizes in real-time, enhancing decision
making in contexts where crowd control is necessary, such as in urban 
transportation systems or large public events. 
6.2 CONTRIBUTIONS OF THE STUDY 
This section highlights the unique contributions of the research and its impact 
on the field of crowd detection and similar applications. 
• Novel Approach: The use of a YOLOv5-based model for crowd detection 
in this project provides a novel application of deep learning techniques. 
This approach offers a high level of accuracy and speed, which is 
particularly important for real-time applications in safety-critical 
environments. 
• Methodological Advances: The study contributes to the field of computer 
vision by demonstrating how existing object detection models can be 
adapted for crowd counting. YOLOv5 was chosen for its speed and 
accuracy, making it suitable for applications where quick analysis is 
crucial. 
• Applications and Use Cases: Potential applications for this research 
include real-time crowd density monitoring for urban transport systems 
(e.g., tracking the number of passengers on buses), large-scale event 
management (e.g., ensuring crowd safety at concerts or sports events), 
and public safety efforts (e.g., identifying overcrowded areas in public 
spaces). 
6.3 LIMITATIONS OF THE STUDY 
In this section, acknowledge the limitations encountered during the research and 
how they impact the findings. 
• Data Limitations: One key limitation of the study was related to the 
quality and variety of data. While datasets like CrowdHuman and COCO 
provided valuable annotated images, the model’s performance may not 
generalize perfectly to real-world scenarios with unique crowd dynamics 
or unconventional settings. 
• Model Limitations: Despite its accuracy, the model struggles with 
extreme crowd densities or occlusions, where people might be partially or 
fully obscured. The model's reliance on visual input also limits its 
effectiveness in low-light conditions or when the camera's resolution is 
too low to capture fine details. 
• Interpretability: As with many deep learning models, the interpretability 
of YOLOv5 remains challenging. The model is often considered a "black 
box," which means it can be difficult to explain exactly why it classifies a 
scene as containing a specific number of people. 
6.4 FUTURE WORK 
This section explores possible directions for future research to address 
limitations and extend the scope of the study. 
• Data Enhancement: Future work should focus on expanding the dataset to 
include more diverse images, such as different lighting conditions, varied 
crowd types, and non-standard environments. This will help the model 
generalize better to real-world scenarios. 
• Model Improvement: Experimenting with advanced architectures, such as 
Transformer-based models (e.g., DETR) or multi-task learning, could 
help improve the model's ability to capture complex crowd structures and 
handle occlusions more effectively. 
• Real-Time Application: In future work, efforts should be made to 
optimize the model for real-time crowd monitoring. Techniques such as 
model compression or quantization could reduce latency and improve the 
feasibility of deploying the model in applications that require fast 
predictions, like live bus tracking. 
• Multimodal Integration: Combining the crowd detection model with other 
data sources, such as infrared sensors or biometric data, could improve 
the system's ability to assess crowd conditions more accurately, especially 
in scenarios with limited visibility or sensor limitations. 
• Explainability and Transparency: Future research should focus on 
improving the explainability of the model’s predictions, particularly for 
applications like public safety. Developing techniques to highlight the 
areas of the image that contribute to the crowd count or creating user
friendly visualization tools could increase trust and adoption of the 
technology. 
6.5 CLOSING REMARKS 
The final section provides a conclusive statement on the overall contributions 
and impact of the research. 
• Project Impact: This research demonstrates the potential of using 
computer vision models to tackle the problem of crowd detection. It 
offers a scalable, efficient, and non-invasive solution for applications 
ranging from urban transportation to large-scale public events, where 
understanding crowd dynamics is critical. 
• Long-Term Vision: Looking ahead, the integration of advanced 
techniques in machine learning and computer vision could pave the way 
for more intelligent systems that automatically manage crowd safety and 
optimize the flow of people in busy urban spaces. This research could 
inspire future innovations in both crowd management and real-time 
public safety technologies. 
