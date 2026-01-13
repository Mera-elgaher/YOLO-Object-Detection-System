from yolo_detector import YOLOv5Detector
import cv2
import urllib.request
import os

def main():
    print("YOLO Object Detection Example")
    print("=" * 40)
    
    # Initialize detector
    print("Loading YOLOv5 model...")
    detector = YOLOv5Detector(model_size='yolov5s', confidence_threshold=0.5)
    
    print(" YOLO detector ready!")
    
    # Download a sample image if it doesn't exist
    sample_image = 'sample_image.jpg'
    if not os.path.exists(sample_image):
        print(" Downloading sample image...")
        url = "https://ultralytics.com/images/bus.jpg"
        try:
            urllib.request.urlretrieve(url, sample_image)
            print(" Sample image downloaded!")
        except:
            print(" Could not download sample image")
            print(" Please add your own image and update the path")
            return
    
    # Run detection
    print(f" Detecting objects in {sample_image}...")
    image, detections, results = detector.detect_image(sample_image)
    
    if detections:
        print(f" Found {len(detections)} objects!")
        
        # Show detection results
        detector.analyze_detections(detections)
        
        # Plot results
        detector.plot_detections(image, detections)
        
        # Save results
        detector.save_results(image, detections, 'detection_result.jpg')
        print(" Results saved to 'detection_result.jpg'")
        
    else:
        print(" No objects detected")
    
    print("\n YOLO detector is ready for:")
    print("- Real-time webcam detection")
    print("- Video processing")
    print("- Batch image processing")
    print("- Custom model training")

if __name__ == "__main__":
    main()
