# WCE Bleeding Region Classification & Detection

## Problem Statement :

- GI bleeding is a medical condition characterised by bleeding in the digestive tract, which circumscribes oesophagus, stomach, small intestine, large intestine (colon), rectum, and anus. Since blood flows into the GI tract, a cascade of risks emerge, ranging from immediate dangers to potential long-term consequences. Excessive blood loss from GI bleeding may lead to a drop in blood pressure, reduced oxygen delivery to organs and tissues, and potentially life-threatening organ dysfunction
- According to World Health Organisation (WHO), gastrointestinal bleeding is responsible for approximately 300,000 deaths every year globally
- WCE Stands for ****************************************************Wireless Capsule Endoscopy**************************************************** , a technique used to analyse GI
- During 6-8 hours of WCE procedure, a video of the GI tract trajectory is recorded on a device attached to the patient’s belt which produces about 57,000-1,00,000 frames; analysed posterior by experienced health experts. Presently, an experienced gastroenterologist takes approximately 2−3 hours to inspect the captured video of one-patient through a frame-by-frame analysis which is not only time-consuming but also susceptible to human error.

## Motivation :

- To come up with a system to find patterns in the bleeding frames, so that it can be used to detect bleeding frames in future patients. This should make the procedure more accessible and making the process less time consuming and less prone to error.
- For Pattern Recognition, Deep Learning is employed for both classification of bleeding and non bleeding frames and detection of bleeding regions in the bleeding frames
    - Combination of Convolutional Networks to extract features from the frames and then using fully connected layers to predict and learn if it is bleeding or non bleeding frame
    - Using Two Stage object detection algorithms for detecting bleeding regions

## Data Preparation :

- The data we have received consists of 2 files :
    - Bleeding file : Contains Bleeding frames and bounding box files
    - Non Bleeding file : Contains Non Bleeding frames
- Code present in the Repository

### Classification :

- Our goal is to redistribute the data in a format accepted by ImageFolder method in the PyTorch datasets library, which is :

```markdown
WCEBleedGen/ <- Overall Dataset Folder
	train/ <- Containing the training Dataset
		Bleeding/ <- Containing the training Bleeding Frames only
		Non Bleeding/ <- Containing the training Non Bleeding Frames only
	test/ <- Containing the test Dataset
		Bleeding/ <- Containing the test Bleeding Frames only
		Non Bleeding/ <- Containing the test Non Bleeding Frames only
	
```

- Approach (overview)
    - For this we first extract all the bleeding and non bleeding images from their respective folders using Path.glob function
    - Then separately we apply train_test_split functionality on the 2 types of data
    - Then we merge training data of bleeding and non bleeding data into one file (train file) and test data of bleeding and non bleeding data into another (test file)
    - within these files we make 2 sub files bleeding and non bleeding

### Detection :

- Our goal is to create a file in the following format using the Bleeding images only :

```markdown
WCEBleedGen/ <- Overall Dataset Folder
	train/ <- Containing the training Dataset
		Images/ <- Containing the training Bleeding Frames only
		Bounding Boxes/ <- Containing the Bounding Boxes of images in order
	test/ <- Containing the test Dataset
		Images/ <- Containing the test Bleeding Frames only
		Bounding Boxes/ <- Containing the Bounding Boxes of images in order
```

- Extract bleeding images and bounding boxes using Path.glob() function
- The Problem we face here is that the files retrieved are not in order (images being mapped to different bounding boxes). To tackle this, we sort the list of file names based on the image number contained in them. This ensures data consistency

## Sample Data Images :

- Bleeding
    
    ![Screenshot 2023-10-24 at 3.38.49 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_3.38.49_PM.png)
    

- Non Bleeding
    
    ![Screenshot 2023-10-24 at 3.39.22 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_3.39.22_PM.png)
    

## Classification :

### Model Architecture :

![Pannu, H.S., Ahuja, S., Dang, N. *et al.* Deep learning based image classification for intestinal hemorrhage. *Multimed Tools Appl* **79**, 21941–21966 (2020). https://doi.org/10.1007/s11042-020-08905-7](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_3.44.41_PM.png)

Pannu, H.S., Ahuja, S., Dang, N. *et al.* Deep learning based image classification for intestinal hemorrhage. *Multimed Tools Appl* **79**, 21941–21966 (2020). https://doi.org/10.1007/s11042-020-08905-7

### Results :

- Loss and Accuracy Plots :
    
    ![Screenshot 2023-10-24 at 3.47.24 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_3.47.24_PM.png)
    
- Precision, Recall and F1 Score (Test Dataset)
    
    ![Screenshot 2023-10-24 at 3.48.13 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_3.48.13_PM.png)
    

## Detection :

### Additional Data Preparation :

- The data that we have for Bleeding region detection is not in the standard form which the PyTorch Dataset Class and DataLoader Class accepts hence we have created a Custom dataset class. It returns the following data
    - Bleeding Image
    - Corresponding Bounding Box and label in a dictionary
- For the proper functioning of the DataLoader, we need to create a Collate Function which would take the batched data and convert it into a list of images and labels

### Pre trained Model :

- We have decided to use **Two Stage object detection** algorithm for our use case because:
    - Unlike one stage object detection algorithm which prioritise speed, two stage object detection algorithms prioritise accuracy. Accuracy is paramount when dealing with medical applications of Machine Learning
    - Two-stage detectors are usually slower than one-stage detectors, but they usually reach better accuracy. They are often used in the medical domain where classification accuracy is more important than speed
- The Pre trained model we have used is the variant of ************************Faster R-CNN************************ trained on the Resnet dataset with a upper limit of 10 box detections per image
- We have frozen all the layers except for the **Region Proposal Network**, and we have changed the classification layer from detecting 91 classes to 2 classes

### Sample Predictions :

- Test Dataset
- Red → Model Prediction, Green → Actual Bleeding Region

![Screenshot 2023-10-24 at 4.18.49 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_4.18.49_PM.png)

![Screenshot 2023-10-24 at 4.25.14 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_4.25.14_PM.png)

![Screenshot 2023-10-24 at 4.25.41 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_4.25.41_PM.png)

![Screenshot 2023-10-24 at 4.26.16 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_4.26.16_PM.png)

- Validation set Predictions :
- White Boundary → Actual Bleeding Region , Red → Model Predictions

![Screenshot 2023-10-24 at 5.19.23 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_5.19.23_PM.png)

![Screenshot 2023-10-24 at 5.21.16 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_5.21.16_PM.png)

![Screenshot 2023-10-24 at 5.22.21 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_5.22.21_PM.png)

![Screenshot 2023-10-24 at 5.24.44 PM.png](WCE%20Bleeding%20Region%20Classification%20&%20Detection/Screenshot_2023-10-24_at_5.24.44_PM.png)

## References :

- Classification Model Architecture :
    - Pannu, H.S., Ahuja, S., Dang, N. *et al.* Deep learning based image classification for intestinal hemorrhage. *Multimed Tools Appl* **79**, 21941–21966 (2020). https://doi.org/10.1007/s11042-020-08905-7
- Detection (Faster R-CNN) :
    - **[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)**
- Dataset :
    - [Palak, Harshita Mangotra, Jyoti Dhatarwaland Nidhi Gooel, ‘WCEbleedGen: A wireless capsule endoscopy dataset containing bleeding and non-bleeding frames’. Zenodo, Jan. 18, 2023. doi: 10.5281/zenodo.7548320.](https://zenodo.org/records/7548320)