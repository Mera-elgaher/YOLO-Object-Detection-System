import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import urllib.request
import os

class YOLOv5Detector:
    def __init__(self, model_size='yolov5s', confidence_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLO detector
        model_size: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        print(f"🔍 Initializing YOLO {model_size} detector...")
        self.load_model()
    
    def load_model(self):
        """Load YOLOv5 model"""
        try:
            # Load model from torch hub
            self.model = torch.hub.load('ultralytics/yolov5', self.model_size, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Set confidence and IoU thresholds
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            print(f"✅ Model {self.model_size} loaded successfully!")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Installing required packages...")
            os.system("pip install ultralytics")
            self.model = torch.hub.load('ultralytics/yolov5', self.model_size, pretrained=True)
    
    def detect_image(self, image_path):
        """Detect objects in a single image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(img_rgb)
            
            # Extract detection data
            detections = results.pandas().xyxy[0].to_dict('records')
            
            return img_rgb, detections, results
            
        except Exception as e:
            print(f"❌ Error detecting objects: {e}")
            return None, None, None
    
    def detect_video(self, video_path, output_path=None):
        """Detect objects in video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_stats = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.model(frame_rgb)
            
            # Draw detections on frame
            annotated_frame = self.draw_detections(frame, results)
            
            # Update statistics
            detections = results.pandas().xyxy[0]
            for class_name in detections['name'].unique():
                if class_name not in detection_stats:
                    detection_stats[class_name] = 0
                detection_stats[class_name] += 1
            
            # Write frame if output specified
            if output_path:
                out.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
            print(f"✅ Video saved to: {output_path}")
        
        return detection_stats
    
    def detect_webcam(self, save_video=False, output_path='webcam_output.mp4'):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Starting webcam detection (Press 'q' to quit)")
        
        # Setup video writer if saving
        if save_video:
            fps = 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Save frame if recording
            if save_video:
                out.write(annotated_frame)
            
            # Display frame
            cv2.imshow('YOLO Object Detection', annotated_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if save_video:
            out.release()
            print(f"✅ Video saved to: {output_path}")
    
    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Create a copy of the image
        img = image.copy()
        
        for _, detection in detections.iterrows():
            # Extract coordinates and info
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Choose color based on class
            color = self.get_class_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
    
    def get_class_color(self, class_name):
        """Get consistent color for each class"""
        # Generate color based on class name hash
        hash_value = hash(class_name) % (256**3)
        color = (
            (hash_value & 0xFF0000) >> 16,
            (hash_value & 0x00FF00) >> 8,
            hash_value & 0x0000FF
        )
        return color
    
    def plot_detections(self, image, detections, figsize=(12, 8)):
        """Plot image with detections using matplotlib"""
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)
        
        for detection in detections:
            # Extract coordinates
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            width, height = x2 - x1, y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection['name']}: {detection['confidence']:.2f}"
            ax.text(x1, y1 - 5, label, bbox=dict(facecolor='red', alpha=0.8), 
                   fontsize=10, color='white')
        
        ax.set_title('Object Detection Results', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def analyze_detections(self, detections):
        """Analyze detection results"""
        if not detections:
            print("No detections found!")
            return
        
        # Count objects by class
        class_counts = {}
        confidence_scores = []
        
        for detection in detections:
            class_name = detection['name']
            confidence = detection['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            confidence_scores.append(confidence)
        
        # Print statistics
        print("📊 Detection Analysis:")
        print(f"Total objects detected: {len(detections)}")
        print(f"Average confidence: {np.mean(confidence_scores):.3f}")
        print(f"Unique classes: {len(class_counts)}")
        
        print("\n📋 Objects detected:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        return class_counts, confidence_scores
    
    def benchmark_performance(self, image_path, num_runs=10):
        """Benchmark detection performance"""
        print(f"⏱️ Benchmarking performance ({num_runs} runs)...")
        
        import time
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            _, detections, _ = self.detect_image(image_path)
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            
            if detections:
                print(f"Run {i+1}: {inference_time:.3f}s - {len(detections)} objects")
        
        # Calculate statistics
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"\n📈 Performance Results:")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Estimated FPS: {fps:.1f}")
        print(f"Min time: {min(times):.3f}s")
        print(f"Max time: {max(times):.3f}s")
        
        return avg_time, fps
    
    def export_to_onnx(self, output_path='yolo_model.onnx'):
        """Export model to ONNX format for deployment"""
        try:
            # Export to ONNX
            self.model.model.export(format='onnx', dynamic=True)
            print(f"✅ Model exported to ONNX format: {output_path}")
        except Exception as e:
            print(f"❌ Error exporting to ONNX: {e}")
    
    def save_results(self, image, detections, output_path):
        """Save detection results"""
        # Draw detections on image
        img_with_detections = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Draw bounding box
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img_with_detections, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save image
        cv2.imwrite(output_path, cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR))
        print(f"💾 Results saved to: {output_path}")

# Custom YOLO training class
class YOLOTrainer:
    def __init__(self, data_yaml_path, model_size='yolov5s'):
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        
    def train_custom_model(self, epochs=100, batch_size=16, img_size=640):
        """Train custom YOLO model"""
        print("🚀 Starting custom YOLO training...")
        
        # Training command
        cmd = f"""
        python train.py --img {img_size} --batch {batch_size} --epochs {epochs} 
        --data {self.data_yaml_path} --weights {self.model_size}.pt --cache
        """
        
        print(f"Training command: {cmd}")
        os.system(cmd)

# Example usage
if __name__ == "__main__":
    print("🔍 YOLO Object Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = YOLOv5Detector(model_size='yolov5s', confidence_threshold=0.5)
    
    # Example: Detect objects in image
    image_path = 'test_image.jpg'
    
    # Download a test image if it doesn't exist
    if not os.path.exists(image_path):
        print("📥 Downloading test image...")
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, image_path)
    
    # Run detection
    image, detections, results = detector.detect_image(image_path)
    
    if detections:
        # Analyze results
        detector.analyze_detections(detections)
        
        # Plot results
        detector.plot_detections(image, detections)
        
        # Save results
        detector.save_results(image, detections, 'detection_result.jpg')
        
        # Benchmark performance
        detector.benchmark_performance(image_path)
        
        print("✅ Object detection completed!")
    else:
        print("❌ No objects detected or error occurred.")
    
    # Example: Real-time webcam detection (uncomment to use)
    # detector.detect_webcam(save_video=True)
    
    print("\n🎯 Detection system ready for deployment!")
