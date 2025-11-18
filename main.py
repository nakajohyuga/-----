from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    print(f"読み込まれたフレーム数: {len(video_frames)}")
    
    #Intialize tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames)
    
    # get object tracks
    object_tracks = tracker.get_object_tracks(video_frames)

    # save video
    save_video(video_frames, "output_videos/output_videos.avi")
    
if __name__ == "__main__":
    main()