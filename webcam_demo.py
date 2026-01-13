from yolo_detector import YOLOv5Detector

def main():
    print("YOLO Webcam Detection Demo")
    print("=" * 40)
    
    # Initialize detector
    detector = YOLOv5Detector(model_size='yolov5s', confidence_threshold=0.5)
    
    print(" Starting webcam detection...")
    print(" Press 'q' to quit")
    
    # Start webcam detection
    detector.detect_webcam(save_video=False)
    
    print("Webcam detection ended!")

if __name__ == "__main__":
    main()
