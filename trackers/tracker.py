from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks={
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)
            
            # Convert Goalkeeper to player object
            for object_ind , class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
                    
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)   
                         
        return tracks
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
             # プレイヤーの描画 (矩形とラベル)
            for track_id, player in player_dict.items():
                 bbox = player["bbox"]
                 x1, y1, x2, y2 = map(int, bbox)
            
                 # チームカラーを取得 (チーム割り当てが完了している場合)
                 color_bgr = player.get('team_color', (255, 0, 0)) # デフォルトは青 (BGR)
                 label = f"P {track_id}"

                # 矩形の描画
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                # ラベルの描画
                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            # 審判の描画 (矩形とラベル)
            for track_id, referee in referee_dict.items():
                 bbox = referee["bbox"]
                 x1, y1, x2, y2 = map(int, bbox)
                 color_bgr = (0, 255, 255) # 黄色 (BGR)
                 label = f"R {track_id}"

                 cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
                # ボールの描画 (矩形とラベル)
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                color_bgr = (0, 255, 0) # 緑 (BGR)
                label = "Ball"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                
            output_video_frames.append(frame)
            
        return output_video_frames