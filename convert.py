import os
from pathlib import Path
import sys
import shutil
from pillow_heif import register_heif_opener
from PIL import Image
import cv2
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Manage Folders
script_dir = Path(__file__).resolve().parent
input_folder_path = Path(f"{script_dir}/Input/")
image_folder_path = Path(f"{script_dir}/ImageFiles/")
depth_folder_path = Path(f"{script_dir}/DepthFiles/")
output_folder_path = Path(f"{script_dir}/Output/")
cleanup_files = False # Careful! Deletes Input Files and Temp Files. Also cleans Output Folder at start.

def checkFolder(folder_path):
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Directory '{folder_path}' created.", file=sys.stdout)
    else:
        print(f"INFO: Delete all Files in: '{folder_path}'", file=sys.stdout)
        for file in folder_path.iterdir():
            if file.is_file():
                file.unlink()

def createPhotoDepth(imgPath):
    file = Path(imgPath)
    from depth_anything_v2.dpt import DepthAnythingV2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'{script_dir}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    raw_img = cv2.imread(imgPath)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    cv2.imwrite(f"{depth_folder_path}/{file.name}".replace(file.suffix,".jpg"), depth)

def createVideoDepth(vidPath):
    file = Path(vidPath)
    from depth_anything_v2.dpt import DepthAnythingV2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'{script_dir}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    raw_video = cv2.VideoCapture(vidPath)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    output_width = frame_width
    output_path = f"{depth_folder_path}/{file.name}".replace(file.suffix,".mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        depth = model.infer_image(raw_frame)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        out.write(depth)
    raw_video.release()
    out.release()

def generateSBS_Photo(img_path, img_gray_path, max_shift=30):
    file = Path(img_path)
    grayfile = Path(img_gray_path)
    if not os.path.exists(img_path) or not os.path.exists(img_gray_path):
        print(f"Missing files for {img_path}")
        return

    image = cv2.imread(img_path)
    depth = cv2.imread(img_gray_path, 0)

    depth = cv2.normalize(depth.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    rows, cols = depth.shape
    left_view = np.zeros_like(image)
    right_view = np.zeros_like(image)

    for y in range(rows):
        for x in range(cols):
            shift = int(depth[y, x] * max_shift)
            if x - shift >= 0:
                left_view[y, x] = image[y, x - shift]
            if x + shift < cols:
                right_view[y, x] = image[y, x + shift]

    sbs_image = np.hstack((left_view, right_view))
    output_path = (f"{output_folder_path}/{file.name}").replace(file.suffix,"_fullsbs.jpg") # https://skybox.xyz/support/How-to-Adjust-2D&3D&VR-Video-Formats
    cv2.imwrite(output_path, sbs_image)
    print(f"Saved: {output_path}")
    if cleanup_files == True:
        file.unlink()
        grayfile.unlink()

def generateSBS_Video(vid_path, vid_gray_path, max_shift=30):
    file = Path(vid_path)
    grayfile = Path(vid_gray_path)
    output_path = (f"{output_folder_path}/{file.name}").replace(file.suffix,"_fullsbs.mp4") # https://skybox.xyz/support/How-to-Adjust-2D&3D&VR-Video-Formats
    if not os.path.exists(vid_path) or not os.path.exists(vid_gray_path):
        print(f"Missing files for {vid_path}")
        return

    # Open video streams
    video_cap = cv2.VideoCapture(vid_path)
    depth_cap = cv2.VideoCapture(vid_gray_path)

    # Get video properties
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    while True:
        ret_img, frame = video_cap.read()
        ret_depth, depth = depth_cap.read()

        if not ret_img or not ret_depth:
            break

        # Convert depth to grayscale and normalize
        depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth_norm = cv2.normalize(depth_gray.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Create left and right views
        left_view = np.zeros_like(frame)
        right_view = np.zeros_like(frame)

        for y in range(height):
            for x in range(width):
                shift = int(depth_norm[y, x] * max_shift)
                if x - shift >= 0:
                    left_view[y, x] = frame[y, x - shift]
                if x + shift < width:
                    right_view[y, x] = frame[y, x + shift]

        # Combine horizontally
        sbs_frame = np.hstack((left_view, right_view))
        out.write(sbs_frame)

    # Release resources
    video_cap.release()
    depth_cap.release()
    out.release()
    print(f"Saved: {output_path}")
    if cleanup_files == True:
        file.unlink()
        grayfile.unlink()

def main():
    if not input_folder_path.is_dir():
        print(f"Error: Directory '{input_folder_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    checkFolder(image_folder_path)
    checkFolder(depth_folder_path)
    if cleanup_files == True:
        checkFolder(output_folder_path)
    if not output_folder_path.is_dir():
        output_folder_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Directory '{output_folder_path}' created.", file=sys.stdout)

    # Extract HEIC File Infos
    register_heif_opener() # Register HEIC support for pillow
    target_suffix = ".heic"
    for file in input_folder_path.iterdir():
        if file.is_file() and file.suffix.lower() == target_suffix.lower():
            image = Image.open(file)
            width, height = image.size
            if image.info["depth_images"]: # When no Depth Information is present we want the file to stay stay so we can generate a Depth Map for it
                print(f"Found file with Depth Information: {file.name}")
                depth_image = image.info["depth_images"][0].to_pillow()
                depth_image = depth_image.resize((width, height)) # Same size as original Image so we dont have to to do that in the sbs generation
                depth_image.save((f"{depth_folder_path}/{file.name}".replace(file.suffix,".jpg")), "JPEG") #Save Depth Image
                image.save((f"{image_folder_path}/{file.name}".replace(file.suffix,".jpg")), "JPEG") #Save Primary Image
                if cleanup_files == True:
                    file.unlink()
            else:
                # heic file convert to jpeg so we can read it later with cv2
                image.save((f"{input_folder_path}/{file.name}".replace(file.suffix,".jpg")), "JPEG") #Save Primary Image
                if cleanup_files == True:
                    file.unlink()

    # Gernate photo maps
    target_suffixes = (".jpg", ".jpeg", ".png")  # Add as many as you need
    for file in input_folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in map(str.lower, target_suffixes):
            print(f"Generate Depth Map for {file.name}")
            createPhotoDepth(file)
            shutil.copy2(file, f"{image_folder_path}/{file.name}")
            if cleanup_files == True:
                file.unlink()
    print("INFO: Start Video Depth analysis this may take some Time!")
    target_suffixes = (".mp4", ".mov")  # Add as many as you need
    for file in input_folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in map(str.lower, target_suffixes):
            print(f"Generate Depth Map for {file.name}")
            createVideoDepth(file)
            shutil.copy2(file, f"{image_folder_path}/{file.name}")
            if cleanup_files == True:
                file.unlink()
    
    target_suffixes = (".jpg", ".jpeg", ".png")  # Add as many as you need
    file_pairs = []
    for file in image_folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in map(str.lower, target_suffixes):
            file_pairs.append((file, (f"{depth_folder_path}/{file.name}").replace(file.suffix,".jpg")))

    print("INFO: Start SBS Photo Generation (max 10 at a time)")
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(run_generate_sbs_photo, file_pairs)
    
    target_suffixes = (".mp4", ".mov")  # Add as many as you need
    file_pairs = []
    for file in image_folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in map(str.lower, target_suffixes):
            file_pairs.append((file, (f"{depth_folder_path}/{file.name}").replace(file.suffix,".mp4")))

    print("INFO: Start SBS Video Generation (max 2 at a time)")
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(run_generate_sbs_video, file_pairs)

def run_generate_sbs_photo(args):
    return generateSBS_Photo(*args)

def run_generate_sbs_video(args):
    return generateSBS_Video(*args)

if __name__ == "__main__":
    main()