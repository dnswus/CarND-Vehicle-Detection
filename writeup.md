## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_car_noncar]: ./images/car_noncar.png
[image_car_hog]: ./images/car_hog.png
[image_noncar_hog]: ./images/noncar_hog.png
[image_search_windows]: ./images/search_windows.png
[image_outcome]: ./images/outcome.png

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In code cell 2 of the Jupyter notebook, I defined a method `get_hog_features` which uses `skimage.feature.hog()` to calculate the HOG of an image. Then in code cell 4, I picked 3 samples of car images and 3 samples of non-car images:

![alt text][image_car_noncar]

In code cell 5 and 6, I explored differrent color spaces and different `skimage.feature.hog()` parameters (`orient`, `pix_per_cell`, and `cell_per_block`). Here are some examples of the car and non-car classes with `color_space=YUV`, `orient=9`, `pix_per_cell=8`, `cell_per_block=2`:

##### Car HOG:
![alt text][image_car_hog]

##### Non-car HOG:
![alt text][image_noncar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on these HOG parameters because the HOG of all 3 channels of the color space can show the original car shape in them.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell 7, I defined 2 helper functions, `img_features` and `extract_features` which helps convert color spaces and extract HOG, spatial binning, and histogram of colors. Then in code cell 8, I used the training data of car and non-car data to train an SVM using LinearSVC.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In code cell 11, I defined `slide_window` which outputs a list of windows based on:
- `x_start_stop` (horizontal start and stop positions)
- `y_start_stop` (vertical start and stop positions)
- `xy_window` (window size)
- `xy_overlap` (overlap percentage)

In code cell 12, I defined `search_windows` and `single_img_features` which scan through specified windows and make predictions for each window to detect car in the images.

In code cell 13, I defined the windows used for the searching. I used smaller windows on the upper part of the road because cars are smaller at that position. And used larger windows toward the bottom part of the road because cars at that closer and appears larger.

![alt text][image_search_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After the car detection, there are some false positives and I used heatmap to eliminate them. Here are the heatmap and final outcome of the test images:

![alt text][image_outcome]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the positive detections I created a heatmap and then thresholded (with threshold of 10) that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on them:

![alt text][image_outcome]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I encounter is when a car is far away, near the top of the road. It is very small and the searching window may only detect it once. However, because of the use of threshold, they are likely to be filtered out. Therefore, `test3.jpg` of the test images cannot detect the white car in it. So cars far away will like be missed in my pipeline. This also applies to cars near the edge of the image because there are fewer windows near the edge.
