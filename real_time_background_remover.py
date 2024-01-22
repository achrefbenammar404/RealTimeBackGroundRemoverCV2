import numpy as np
import mediapipe as mp
import cv2


class BackGroundRemover() : 
    
    
    BG_COLOR = (255, 255, 255) # White 
    MASK_COLOR = (0, 0, 0) # Black
    
    model_path = 'bg_remove/selfie_segmenter_landscape.tflite'
    BaseOptions = mp.tasks.BaseOptions
    base_options = BaseOptions(model_asset_path=model_path)
    
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                output_category_mask=True )  
    @staticmethod
    def __call__(**inputs) : 
        """The Image Segmenter outputs a list of Image data. 
        If output_type is CATEGORY_MASK, the output is a list 
        containing single segmented mask as an uint8 image. The pixel 
        indicates the recognized category index of the input image. 
        If output_type is CONFIDENCE_MASK, the output is a vector with size of 
        category number. Each segmented mask is a float image within the range [0,1], 
        representing the confidence score of the pixel belonging to the category.
        """
        with BackGroundRemover.ImageSegmenter.create_from_options(BackGroundRemover.options) as segmenter:
            #inputs -> numpy_array 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inputs['image'])
            segmented_masks = segmenter.segment(mp_image)
            category_mask = segmented_masks.category_mask

            # Generate solid color images for showing the output segmentation mask.
            image_data = mp_image.numpy_view()
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BackGroundRemover.BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.9
            output_image = np.where(condition, bg_image, image_data)

            return output_image
            
            
    



#How to use 
import cv2

# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Livestream', BackGroundRemover()(image = (frame)))

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
