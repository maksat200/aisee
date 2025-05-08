import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO
from collections import defaultdict

class CustomerCounter:
    def __init__(self, video_path, confidence=0.5, line_position=0.5, show_video=True, save_video=False):
        """
        Initialize the customer counter
        
        Args:
            video_path: Path to the video file
            confidence: Detection confidence threshold
            line_position: Position of the counting line (0-1, relative to video height)
            show_video: Whether to display the video
            save_video: Whether to save the processed video
        """
        self.video_path = video_path
        self.confidence = confidence
        self.show_video = show_video
        self.save_video = save_video
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate line position
        self.line_y = int(self.frame_height * line_position)
        
        # Initialize counters
        self.entry_count = 0
        self.exit_count = 0
        
        # Initialize tracking
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store tracked objects {id: (centroid, consecutive_misses)}
        self.disappeared = {}  # Dictionary to track consecutive frames an object has been missing
        self.crossed_line_ids = set()  # Set to store IDs that have crossed the line
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using the nano model for speed
        
        # Initialize video writer if saving
        self.out = None
        if self.save_video:
            output_path = video_path.rsplit('.', 1)[0] + '_counted.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                      (self.frame_width, self.frame_height))
    
    def process_frame(self, frame):
        """Process a single frame to detect and track people"""
        # Run YOLOv8 inference on the frame
        results = self.model(frame, classes=0)  # Class 0 is person in COCO dataset
        
        # Extract person detections
        current_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                if conf > self.confidence:
                    # Calculate centroid
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    
                    # Add to current detections
                    current_detections.append((int(x1), int(y1), int(x2), int(y2), centroid_x, centroid_y))
        
        # Update tracking
        self.update_tracking(current_detections)
        
        # Draw line
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), (0, 255, 255), 2)
        
        # Draw detections and tracking info
        for obj_id, (centroid, _) in self.objects.items():
            # Draw centroid
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
            
            # Draw ID
            text = f"ID: {obj_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw counter information
        info_text = f"Entries: {self.entry_count} | Exits: {self.exit_count} | Current: {self.entry_count - self.exit_count}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def update_tracking(self, detections):
        """Update object tracking based on new detections"""
        # If we have no objects, register all detections as new objects
        if len(self.objects) == 0:
            for x1, y1, x2, y2, cx, cy in detections:
                self.register_object((cx, cy))
            return
        
        # If we have no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                
                # If object has disappeared for too many frames, deregister it
                if self.disappeared[obj_id] > 20:  # Adjust this threshold as needed
                    self.deregister_object(obj_id)
            return
        
        # Calculate distances between existing objects and new detections
        object_centroids = [centroid for centroid, _ in self.objects.values()]
        detection_centroids = [(cx, cy) for _, _, _, _, cx, cy in detections]
        
        # Calculate distance matrix
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                distances[i, j] = np.sqrt((obj_centroid[0] - det_centroid[0])**2 + 
                                         (obj_centroid[1] - det_centroid[1])**2)
        
        # Find minimum distance assignments
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        # Keep track of which rows and columns we've already examined
        used_rows = set()
        used_cols = set()
        
        # Loop over the combinations
        for (row, col) in zip(rows, cols):
            # If we've already examined this row or column, skip it
            if row in used_rows or col in used_cols:
                continue
            
            # If the distance is too large, skip it
            if distances[row, col] > 50:  # Adjust this threshold as needed
                continue
            
            # Get the object ID for this row
            obj_id = list(self.objects.keys())[row]
            
            # Update the centroid
            centroid = detection_centroids[col]
            self.objects[obj_id] = (centroid, self.objects[obj_id][1])
            
            # Reset the disappeared counter
            self.disappeared[obj_id] = 0
            
            # Check if object crossed the line
            prev_centroid = self.objects[obj_id][1]
            if prev_centroid is not None:
                if obj_id not in self.crossed_line_ids:
                    # Check if the object crossed the line from top to bottom
                    if prev_centroid[1] < self.line_y and centroid[1] >= self.line_y:
                        self.entry_count += 1
                        self.crossed_line_ids.add(obj_id)
                    # Check if the object crossed the line from bottom to top
                    elif prev_centroid[1] >= self.line_y and centroid[1] < self.line_y:
                        self.exit_count += 1
                        self.crossed_line_ids.add(obj_id)
            
            # Update previous centroid
            self.objects[obj_id] = (centroid, centroid)
            
            # Mark row and column as used
            used_rows.add(row)
            used_cols.add(col)
        
        # Handle unused rows (disappeared objects)
        unused_rows = set(range(distances.shape[0])) - used_rows
        for row in unused_rows:
            obj_id = list(self.objects.keys())[row]
            self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
            
            # If object has disappeared for too many frames, deregister it
            if self.disappeared[obj_id] > 20:  # Adjust this threshold as needed
                self.deregister_object(obj_id)
        
        # Handle unused columns (new objects)
        unused_cols = set(range(distances.shape[1])) - used_cols
        for col in unused_cols:
            self.register_object(detection_centroids[col])
    
    def register_object(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = (centroid, None)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister_object(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def run(self):
        """Process the entire video"""
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            if self.show_video:
                cv2.imshow('Customer Counter', processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save the frame
            if self.save_video and self.out is not None:
                self.out.write(processed_frame)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Processed {frame_count} frames. FPS: {fps:.2f}")
        
        # Release resources
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print(f"\nFinal Results:")
        print(f"Total Entries: {self.entry_count}")
        print(f"Total Exits: {self.exit_count}")
        print(f"Current Count: {self.entry_count - self.exit_count}")
        
        return {
            "entries": self.entry_count,
            "exits": self.exit_count,
            "current": self.entry_count - self.exit_count
        }


def main():
    parser = argparse.ArgumentParser(description='Count customers in a video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--line-position', type=float, default=0.5, 
                        help='Position of the counting line (0-1, relative to video height)')
    parser.add_argument('--no-display', action='store_true', help='Do not display the video')
    parser.add_argument('--save', action='store_true', help='Save the processed video')
    
    args = parser.parse_args()
    
    counter = CustomerCounter(
        video_path=args.video_path,
        confidence=args.confidence,
        line_position=args.line_position,
        show_video=not args.no_display,
        save_video=args.save
    )
    
    results = counter.run()
    return results


if __name__ == "__main__":
    main()
