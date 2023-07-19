import os


def valid_folder(folder_save_cropped: str):
    """
    process for create a folder if not exist

    @type folder_save_cropped: str
    @param folder_save_cropped: path to folder to create if not exist
    """
    os.makedirs(folder_save_cropped, exist_ok=True)
    for file in os.listdir(folder_save_cropped):
        try:
            os.remove(os.path.join(folder_save_cropped, file))
        except:
            try:
                os.rmdir(os.path.join(folder_save_cropped, file))
            except:
                pass

def create_data_using_video(path_video:str) -> List[dict]:
    """
    process for load all frames of video

    @type kwargs: str
    @param kwargs: path to video

    @rtype: list
    @returns: list with each frame in dict format
    """
    data_frames = []

    import tempfile, os
    temp_dir = tempfile.gettempdir()
    path_save_frames = os.path.join(temp_dir, "video")
    os.makedirs(path_save_frames, exist_ok=True)

    import cv2
    cap = cv2.VideoCapture(path_video)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(path_save_frames, f"{count}.jpg"), frame)
            data_frames.append({
                "source": os.path.join(path_save_frames, f"{count}.jpg"),
            })
            count += 1
        else:
            break
    cap.release()