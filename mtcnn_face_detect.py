from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import List, Tuple
import shutil

import ffmpeg
from PIL import Image
import numpy as np
from tqdm import tqdm

from facenet_pytorch import MTCNN

try:
    set_start_method("spawn")
except RuntimeError:
    pass

mtcnn: MTCNN  = None


def extract_face(image: Image.Image, box: List[int], save_path: Path) -> None:
    img = np.array(image)
    box = [
        int(max(box[0], 0)),
        int(max(box[1], 0)),
        int(min(box[2], img.shape[1])),
        int(min(box[3], img.shape[0])),
    ]
    cropped_image = img[box[1] : box[3], box[0] : box[2]]
    cropped_image = Image.fromarray(cropped_image)
    cropped_image.save(save_path)


def face_detect(file: str, save_file: str) -> None:
    """
    Detects faces in an image using MTCNN.
    If no faces are detected, the image is not saved.
    If multiple faces are detected, the largest face is used.
    The face is cropped and saved to the output directory.
    """
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN(post_process=False, device="cuda")

    img = Image.open(file)
    boxes, probs, points = mtcnn.detect(img, landmarks=True)  # type: ignore
    if boxes is None or len(boxes) == 0:
        return
    extract_face(img, boxes[0].squeeze().tolist(), save_path=save_file)


def get_pairs(rdir: str, odir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    files = Path(rdir).rglob("*")
    for i, file in enumerate(files):
        if file.suffix.lower() not in [".png"]:
            continue
        ofile = Path(str(file).replace(rdir, odir))
        ofile.parent.mkdir(parents=True, exist_ok=True)
        if ofile.exists():
            continue
        pairs.append((str(file), str(ofile)))
    return pairs


def extract_frames(video_path: Path, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(save_dir / "frame_%05d.png")

    try:
        (
            ffmpeg.input(str(video_path))
            .output(output_pattern, **{"qscale:v": 1})
            .run(quiet=True, overwrite_output=True)
        )
        # print(f"✅ Frames extracted from {video_path}")
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg error on {video_path}: {e}")


def extract_frames_wrapper(args):
    return extract_frames(*args)


def extract_frames_for_dataset(rdir: str, frames_dir: str) -> None:
    args = []
    for video_path in Path(rdir).rglob("*"):
        if video_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
            continue
        output_dir = Path(
            str(video_path).replace(rdir, frames_dir).replace(video_path.suffix, "")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        args.append((video_path, output_dir))

    with Pool(8) as p:
        _ = list(tqdm(p.imap(extract_frames_wrapper, args), total=len(args)))


def face_detect_wrapper(args):
    return face_detect(*args)


def casia_restructure() -> None:
    rdir = Path("/mnt/cluster/nbl-datasets/spoof-face/Spoof_Database_China_CBSR/")
    odir = Path("/mnt/cluster/nbl-datasets/spoof-face/CASIA_FASD/raw")
    odir.mkdir(parents=True, exist_ok=True)
    for ssplit_dir in rdir.iterdir():
        if not ssplit_dir.is_dir():
            continue
        for sub_dir in tqdm(ssplit_dir.iterdir(), desc=ssplit_dir.name):
            if not sub_dir.is_dir():
                continue
            try:
                int(sub_dir.name)
            except ValueError:
                continue
            # Low quality
            subject = ssplit_dir.name + "_" + sub_dir.name

            bon_video_path = sub_dir / "1.avi"
            obon_video_path = odir / "Bonafide" / subject / "1.avi"
            obon_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bon_video_path, obon_video_path)

            print_video_path = sub_dir / "3.avi"
            oprint_video_path = odir / "Attack" / "Print" / subject / "3.avi"
            oprint_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(print_video_path, oprint_video_path)

            cut_video_path = sub_dir / "5.avi"
            ocut_video_path = odir / "Attack" / "Cut" / subject / "5.avi"
            ocut_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cut_video_path, ocut_video_path)

            screen_video_path = sub_dir / "7.avi"
            oscreen_video_path = odir / "Attack" / "Screen" / subject / "7.avi"
            oscreen_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(screen_video_path, oscreen_video_path)

            # Medium quality
            bon_video_path = sub_dir / "2.avi"
            obon_video_path = odir / "Bonafide" / subject / "2.avi"
            obon_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bon_video_path, obon_video_path)

            print_video_path = sub_dir / "4.avi"
            oprint_video_path = odir / "Attack" / "Print" / subject / "4.avi"
            oprint_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(print_video_path, oprint_video_path)

            cut_video_path = sub_dir / "6.avi"
            ocut_video_path = odir / "Attack" / "Cut" / subject / "6.avi"
            ocut_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cut_video_path, ocut_video_path)

            screen_video_path = sub_dir / "8.avi"
            oscreen_video_path = odir / "Attack" / "Screen" / subject / "8.avi"
            oscreen_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(screen_video_path, oscreen_video_path)

            # High quality
            bon_video_path = sub_dir / "HR_1.avi"
            obon_video_path = odir / "Bonafide" / subject / "HR_1.avi"
            obon_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bon_video_path, obon_video_path)

            print_video_path = sub_dir / "HR_2.avi"
            oprint_video_path = odir / "Attack" / "Print" / subject / "HR_2.avi"
            oprint_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(print_video_path, oprint_video_path)

            cut_video_path = sub_dir / "HR_3.avi"
            ocut_video_path = odir / "Attack" / "Cut" / subject / "HR_3.avi"
            ocut_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cut_video_path, ocut_video_path)

            screen_video_path = sub_dir / "HR_4.avi"
            oscreen_video_path = odir / "Attack" / "Screen" / subject / "HR_4.avi"
            oscreen_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(screen_video_path, oscreen_video_path)


def driver(frames_dir: str, facedetect_dir: str, num_process: int) -> None:
    pairs = get_pairs(frames_dir, facedetect_dir)
    with Pool(num_process) as p:
        _ = list(tqdm(p.imap(face_detect_wrapper, pairs), total=len(pairs)))

if __name__ == "__main__":
    rdir = "/mnt/cluster/nbl-datasets/spoof-face/CASIA_FASD/raw"
    frames_dir = "/mnt/cluster/nbl-datasets/spoof-face/CASIA_FASD/frames"
    facedetect_dir = "/mnt/cluster/nbl-datasets/spoof-face/CASIA_FASD/facedetect"
    # extract_frames_for_dataset(rdir, frames_dir)
    # idiap_replay_extract_frames(rdir, frames_dir)
    driver(frames_dir, facedetect_dir, 16)
