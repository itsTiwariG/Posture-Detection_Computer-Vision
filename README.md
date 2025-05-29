
Course Project: Posture Detection in Videos

Computer Vision (CSL 7360)
Adarsh Tiwari
April 2025
GitHub | Website


Introduction

Posture detection in videos is a key problem in computer vision with applications in healthcare, fitness, workplace safety, and smart surveillance. It involves identifying and tracking human body positions across video frames to understand how a person is standing, sitting, or moving.
In this project, we explore several computer vision and deep learning methods to detect posture accurately and efficiently. We experimented with models like PoseNet, OpenPose, MediaPipe PoseDetector, and BlazePose, which are popular deep learning-based pose estimation tools. We also used traditional techniques such as a CNN-based posture classification model and the Lucas-Kanade optical flow algorithm for motion tracking.

Each method has its own strengths, some are better at real-time performance, while others provide more detailed body keypoints. Our goal is to compare these methods and build a robust system that can detect posture effectively in different video conditions.


Posture Detection: An overview
Posture detection in video refers to the process of identifying and tracking the positions of human body joints (key points) across a sequence of frames. The task usually begins with detecting the person in each frame, either using bounding boxes or full-frame analysis.

Once the person is localized, pose estimation models are applied to predict the 2D coordinates of keypoints such as the head, shoulders, elbows, hips, knees, and ankles. Each keypoint is also assigned a confidence score to reflect how certain the model is about its prediction.

These key points can be detected in each frame independently or tracked across time to capture motion and posture transitions. The extracted posture data can then be used to understand human activities, identify incorrect postures, or analyze movement patterns.


Methods
In this project, we explored various methods for human posture detection in videos, from traditional computer vision techniques to deep learning-based models. Each method offers different trade-offs in accuracy, speed, and use-case suitability. We began with the Lucas-Kanade Optical Flow, a lightweight method that tracks keypoint movement based on pixel intensity changes, effective for small, smooth motions. Next, we implemented a CNN-based posture classifier, which predicts postures directly from video frames without key point extraction, simple and useful for basic tasks. Finally, we used advanced pose estimation models like PoseNet, OpenPose, MediaPipe PoseDetector, and BlazePose, which detect key body joints with confidence scores and enable detailed posture analysis.
Lucas-Kanade Optical Flow
The Lucas-Kanade method is a classical computer vision technique used to estimate the motion of features across frames in a video. For posture detection, this method is applied to track key points such as joints or body parts by analyzing how they move over time.

Assumption: Intensity of a small region around a pixel remains constant between consecutive frames.
Under this assumption, the optical flow is estimated by solving a set of linear equations for each small window in the image.
In our project, we used the Lucas-Kanade algorithm to track body joints initialized manually or from previous pose estimates. It uses image pyramids to handle larger motions by tracking at multiple scales, from coarse to fine resolution.
Pros: It is computationally light and runs in real-time, which makes it suitable for applications where resources are limited.
Cons: It doesn’t work well with videos involving occlusion and small subject-to-background size ratio. One major limitation is that it relies on the assumption of small, consistent motion, which is not always valid in dynamic scenes. Focuses on key features in background, rather than just subject.
Results

Horn-Schunck Optical Flow
The Horn-Schunck method is a classical optical flow algorithm used to estimate the motion of pixels between consecutive video frames. It assumes that the motion of objects in the scene is smooth and continuous, making it suitable for tracking subtle movements like body joint motion.

In the context of posture detection, Horn-Schunck can be used to track keypoints (e.g., elbows, knees) across frames after they are initially identified using a pose estimation model. It computes a dense flow field by minimizing both the difference in pixel intensity and the variation in flow between neighboring pixels, resulting in smooth motion vectors.
Pros: Produces smooth and dense motion tracking. Shows most joints and points correctly. Focuses on localised point detection.
Cons: Sensitive to noise and large movements. Slower and more computationally expensive than sparse methods like Lucas-Kanade. Often fails to detect complete body based on angle/warp.

Results


Traditional CNN-Based Posture Classification

We implemented a CNN-based keypoint prediction model using a modified ResNet-18 backbone. The original fully connected layer of ResNet-18 was removed, and a custom head was added to predict the x and y coordinates of 16 human body key points, resulting in an output layer with 32 nodes. The model was trained on the MPII Human Pose Dataset using Mean Squared Error (MSE) loss and the Adam optimizer. Each training image was paired with a 32-dimensional vector representing the normalized coordinates of body joints such as shoulders, elbows, knees, etc. The model learned to directly regress these key points from raw input images without relying on intermediate heatmaps or keypoint detectors.


MPII Human Pose Dataset

Model Architecture


Pros: Simple model, no post processing required. It is also lightweight and fast, making it suitable for low latency application

Cons: Unlike heatmap-based methods, direct coordinate regression lacks spatial precision, struggles with occlusions, cluttered scenes, or multiple people, and does not provide confidence scores for keypoints.

Results


PoseNet
		
PoseNet is a deep learning-based model designed to estimate human body keypoints from images or videos. It detects 17 key points—including nose, eyes, shoulders, elbows, hips, knees, and ankles—along with confidence scores that indicate the reliability of each prediction. In our project, we used PoseNet for single-person posture tracking across video frames.
PoseNet was originally trained on the COCO Keypoints dataset, which contains a wide variety of human poses in diverse environments. It uses MobileNetV1 as the backbone, a lightweight convolutional neural network optimized for speed and mobile deployment. The architecture produces heatmaps for each keypoint, and the final keypoint locations are extracted by taking the maximum activation in each heatmap.
COCO Keypoints dataset

Model Architecture

Pros: Fast and efficient, even on mobile and low-power devices. Provides confidence scores for each keypoint.
Cons: Performance may degrade in unusual poses or extreme occlusion.
Results

MediaPipe PoseDetector
MediaPipe Pose is a lightweight, real-time pose estimation framework developed by Google. It can detect 33 key points of the human body, including major joints and additional landmarks like the face and feet, offering more detail than models like PoseNet.
It works in a 2 step process:
A person is first detected using a bounding box.
Within that region, the model predicts the precise 3D locations of keypoints along with confidence scores.
The model architecture is optimized for speed and accuracy, making it suitable for use on mobile devices and web applications.
Pros: High accuracy and 3D keypoint estimation. Extremely fast and optimized for real-time use.
Cons: Designed mainly for single-person detection.
Results


BlazePose
BlazePose is a high-performance human pose estimation model developed by Google, designed for real-time applications. It forms the backbone of MediaPipe’s pose estimation pipeline. BlazePose predicts 33 body landmarks, offering detailed and anatomically aligned joint positions, including face, hands, and feet keypoints.
BlazePose uses a two-step pipeline:
A lightweight detector first identifies the region containing the person.
A regression-based model then predicts the precise coordinates of 33 body key points within the detected region.
It improves accuracy by incorporating depth estimation, which gives it limited 3D capability. The model is optimized for speed, using techniques like anchor-free detection and lightweight convolutional layers, making it ideal for mobile and embedded systems.
We built a human keypoint detection model BlazePose using TensorFlow and MobileNetV2 as a frozen backbone. It processes COCO dataset images, extracts and normalizes 17 body keypoints, applies data augmentation, and trains a neural network to predict keypoint positions. The model includes an initial prediction head and a refinement block to improve accuracy. It also provides visualization tools for both keypoints and intermediate feature maps.

Pros: Provides pseudo-3D (depth-aware) key points. Robust to common challenges like partial occlusions or fast motion.
Cons: May lose accuracy in cluttered or multi-person scenes.
Results





Keypoint RCNN
Keypoint R-CNN is an extension of the Faster R-CNN framework for single-person 2D pose estimation, developed by Facebook AI Research (FAIR). It is designed to predict the locations of human keypoints (e.g., joints) along with object detection in images. Unlike bottom-up methods like OpenPose, Keypoint R-CNN uses a top-down approach.
It uses a Region Proposal Network (RPN) to first detect people in the image.
For each person detected, a keypoint head predicts the 2D locations of predefined human keypoints using heatmaps.

The keypoint head is a small Fully Convolutional Network (FCN) that operates on the aligned features extracted from the RoIAlign operation.
The backbone is typically a ResNet-FPN (Feature Pyramid Network), which enables multi-scale feature extraction. Keypoint R-CNN typically estimates 17 COCO keypoints (like nose, shoulders, elbows, etc.).
Pros: Accurate for single person, Robust to occlusions and cluttered backgrounds.
Cons: Slower in multi-person settings. 
Results


Website Screenshot



Contributions
Soham Deshmukh: Lucas-Kanade Optical Flow, MediaPipe PoseDetector, Horn-Schunck Optical Flow.
Akriti Gupta: PoseNet, Website, Report, Research Work.
Sagnik Goswami: CNN-based Posture Classification, Website, Report.
Adarsh Tiwari: Keypoint RCNN,  Research Work, Report.
Rutuja Janbandhu: BlazePose, Research Work, Report.

References
Cao, Z., Hidalgo, G., Simon, T., Wei, S. E., & Sheikh, Y. (2019). OpenPose: Realtime multi-person 2D pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1), 172–186.
Papandreou, G., Zhu, T., Chen, L. C., Gidaris, S., Tompson, J., & Murphy, K. (2018). PersonLab: Person pose estimation and instance segmentation with a bottom-up, part-based, geometric embedding model. Proceedings of the European Conference on Computer Vision (ECCV), 2018.
Toshev, A., & Szegedy, C. (2014). DeepPose: Human pose estimation via deep neural networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Simon, T., Joo, H., Matthews, I., & Sheikh, Y. (2017). Hand keypoint detection in single images using multiview bootstrapping. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C. L., & Grundmann, M. (2020). BlazePose: On-device real-time body pose tracking. arXiv preprint arXiv:2006.10204.
Kreiss, S., Bertoni, L., & Alahi, A. (2019). PifPaf: Composite fields for human pose estimation. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Horn, B. K., & Schunck, B. G. (1981). Determining optical flow. Artificial Intelligence, 17(1-3), 185–203.
Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI).
Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C. (2014). Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(7), 1325–1339.
Andriluka, M., Pishchulin, L., Gehler, P., & Schiele, B. (2014). 2D Human pose estimation: New benchmark and state of the art analysis. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).


