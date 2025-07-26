Alzheimer's Disease Prediction Using Longitudinal Magnetic Resonance Imaging and Deep Learning (DenseNet-BiLSTM)
Abstract
This project presents the implementation of a novel hybrid deep learning framework for Alzheimer's disease prediction using longitudinal MRI images. The primary goal is to overcome common challenges in Alzheimer's diagnosis, such as limited sample size, high data dimensionality, the need to capture temporal dynamics, overfitting, and vanishing gradients. The proposed model is based on combining DenseNet and BiLSTM networks, which simultaneously extract and classify spatial and temporal features from longitudinal MRI images over four years. The model is evaluated on 684 MRI images from the ADNI database, and experimental results show an accuracy of 95.28% in AD/CN classification, significantly improving Alzheimer's diagnosis.

Keywords: Alzheimer's disease, longitudinal analysis, magnetic resonance imaging, DenseNet, BiLSTM, deep learning

Research Innovation
The innovative aspect of this study lies in the development and application of a new hybrid deep learning method for predicting Alzheimer's disease using longitudinal MRI image analysis. This approach proposes a unique DenseNet-BiLSTM framework that effectively extracts and combines spatial and temporal features from sequential neuroimaging data. This innovation addresses several fundamental challenges in Alzheimer's diagnosis, including the need to analyze disease dynamics over time, combat overfitting, solve the vanishing gradient problem, and manage incomplete patient data.

Methodology
1. Data
This project utilizes longitudinal MRI images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) database.

Number of Participants: 684 individuals.

Modality: T1-weighted MRI images with 3 Tesla field strength.

Time Points: Images collected at 7 different time points (Baseline, M6, M12, M18, M24, M36, M48) over 48 months.

Classes: Four main classes are used: Normal Cognition (NC), Stable Mild Cognitive Impairment (sMCI), Progressive Mild Cognitive Impairment (pMCI), and Alzheimer's Disease (AD).

2. Data Preprocessing
Preprocessing steps are crucial to ensure data homogeneity and quality:

Loading: MRI images are loaded in NIfTI format.

Intensity Nonuniformity Correction: Using the N3 algorithm.

Smoothing and AC-PC Correction: Images are smoothed to 256x256x256 dimensions, and AC-PC (Anterior Commissure - Posterior Commissure) correction is performed with MIPAV.

Skull Stripping and Cerebellum Removal: Separation of brain tissue from the skull and removal of the cerebellum.

Tissue Segmentation: Segmentation of MRI images into Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF) using FAST.

Spatial Normalization: Using mass-preserving deformable warping (e.g., HAMMER) for spatial normalization of images from different time points into a standard space (e.g., MNI).

Extraction of Regional Volumetric Maps (RAVENS maps): Focusing on Gray Matter (GM) volumetric maps.

Intensity Normalization: Normalizing the intensity of each image (mean 0, standard deviation 1).

Zero-Density Pixel Removal: Resizing images to 200x168x168 voxels.

Important Note: The code provided in this repository replaces the actual ADNI data preprocessing section with Dummy Data Generation due to its complexity and the need for specialized tools (such as FSL, SimpleITK) and access to raw ADNI data. You will need to replace this section with your actual implementation for ADNI data.

3. Model Architecture (Hybrid DenseNet-BiLSTM)
The proposed model is a hybrid deep learning framework consisting of two main components:

DenseNet for Spatial Feature Extraction:

Purpose: To extract 3D spatial features (patterns, edges, local structures) from each MRI image.

Functionality: DenseNet, with its dense connections, improves information flow and gradients, prevents vanishing gradients, and reduces the number of parameters and the risk of overfitting by reusing features.

Implementation: Multiple (identical) 3D DenseNet instances are used, each applied in a TimeDistributed manner to one image from the time sequence.

BiLSTM for Temporal Feature Extraction:

Purpose: To capture temporal relationships and dependencies between the spatial feature sequences extracted by DenseNet.

Functionality: BiLSTM (Bidirectional Long Short-Term Memory network) can model disease progression dynamics over time from both forward and backward directions, effectively managing the vanishing gradient problem.

Implementation: The outputs from DenseNet are fed as a sequential input to the BiLSTM layer.

Final Classification:

Purpose: To perform the final disease classification (AD/CN, NC/MCI, sMCI/pMCI, MCI/AD).

Functionality: The extracted spatio-temporal features are passed to a fully connected (Dense) layer and then converted into class probabilities using the Softmax activation function.

Optimization: Cross-Entropy Loss function and the Adam optimizer are used for model training.

4. Training and Evaluation
Data Splitting: The dataset is stratified into 90% for training and validation, and 10% for testing.

Cross-Validation: Stratified 10-fold cross-validation is used to determine optimal hyperparameters and evaluate model stability.

Hyperparameters:

Epochs: 90

Batch Size: 64

Learning Rate: 0.001 (for Adam optimizer)

Activation: PReLU (in DenseNet layers), Softmax (output layer)

Regularization: Dropout (with a rate of 0.10) and L2 regularization (with 
lambda=0.01) to prevent overfitting.

Performance Metrics:

Accuracy

Sensitivity / Recall

Specificity

Receiver Operating Characteristic (ROC) Curve

Area Under the ROC Curve (AUC)

How to Run
1. Clone the Repository
git clone https://github.com/AlirezaJomeiri/AlzheimerPrediction-DenseNet-BiLSTM.git
cd AlzheimerPrediction-DenseNet-BiLSTM

2. Install Dependencies
pip install -r requirements.txt

Contents of requirements.txt (may include more):

tensorflow>=2.x
keras>=2.x
numpy
scikit-learn
tqdm # For progress bar display
# For actual MRI processing, the following might be needed:
# nibabel
# SimpleITK
# (and external tools like FSL or FreeSurfer)

3. Prepare Data
This is the most crucial step.

ADNI Data Access: First, you must request access to and download the data from the ADNI website.

Directory Structure: Place your raw MRI data in the data/ADNI_MRI_raw directory (or the path you configure in main.py). The data structure should allow the preprocessing script to locate the files (e.g., data/ADNI_MRI_raw/PatientID/TimePoint/scan.nii.gz).

Implement Actual Preprocessing: As mentioned, the load_and_preprocess_adni_data function in main.py currently generates only dummy data. You must replace this function with the actual logic for loading, correcting, aligning, and segmenting your MRI images. This involves using libraries like nibabel for NIfTI reading and SimpleITK for image processing operations, and potentially calling external FSL or FreeSurfer scripts for more complex steps (like skull stripping and tissue segmentation).

4. Run the Code
After preparing the data and preprocessing configurations, you can run the main script:

python main.py

This command will initiate the process of data loading (or dummy data generation), model construction, training (with cross-validation), and evaluation.

Results
The proposed Hybrid DenseNet & BiLSTM model has shown very promising results in Alzheimer's disease classification:

AD/CN Classification Accuracy: 95.28%

NC/MCI Classification Accuracy: 88.19%

sMCI/pMCI Classification Accuracy: 83.51%

MCI/AD Classification Accuracy: 92.14%

These results demonstrate the superiority of the proposed method over baseline approaches (such as VGG19, ResNet50, DenseNet, and BiLSTM alone) in leveraging spatial and temporal information from longitudinal MRI images for more accurate Alzheimer's diagnosis.

Limitations and Future Work
Limitations
High Training Time and Computational Costs: Training complex deep models like DenseNet-BiLSTM on 3D longitudinal data is time-consuming and requires significant computational resources (GPU).

Reliance on High-Quality MRI Images: The model's performance is heavily dependent on the high quality of input MRI images, which may not be available in all clinical settings.

Suggestions for Future Work
Utilize Multimodal Data: Combining MRI with other modalities such as PET, EEG, clinical, and genetic data to achieve a more comprehensive diagnosis.

Model Optimization: Investigating techniques like model pruning, quantization, or using specialized hardware (e.g., TPUs) to reduce training time and computational costs.

Preprocessing Optimization: Researching more advanced preprocessing techniques (e.g., high-resolution alignment, noise reduction) and using transfer learning to further improve accuracy.

Alternative Architectures: Exploring newer deep learning architectures such as Transformers or Capsule Networks to further mitigate overfitting and vanishing gradient problems.

Contact
For any questions or suggestions, please contact me at jomeiri@yahoo.com

License
This project is released under the MIT License. See the LICENSE file for more details.
