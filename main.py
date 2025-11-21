from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    #Intialize tracker
    tracker = Tracker("models/best.pt")

   
    
    # get object tracks
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="stubs/track_stub.pkl")

    print(f"読み込まれたビデオフレーム数: {len(video_frames)}")
    if 'players' in tracks and tracks['players']:
        print(f"トラッキング結果のリスト長 (players): {len(tracks['players'])}")
    else:
        print("トラッキング結果 (players) が空または欠損しています")
        
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    print("アノテーション描画が完了しました。")
    
    # save video
    save_video(output_video_frames, "output_videos/output_videos.avi")
    
if __name__ == "__main__":
    main()