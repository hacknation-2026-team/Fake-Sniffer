import os
import json
import cv2
import imagehash
from PIL import Image


class LightweightKeyframeExtractor:
    def __init__(
        self,
        sample_interval_sec: float = 0.5,
        min_sharpness: float = 60.0,
        hash_diff_threshold: int = 6,
        max_keyframes: int = 8,
    ):
        """
        :param sample_interval_sec: Co ile sekund pobierać próbkę (0.5s = 2 klatki/s).
        :param min_sharpness: Minimalna ostrość (wariancja Laplasjanu) - odrzuca rozmazane klatki.
        :param hash_diff_threshold: Minimalna odległość Hamminga między hashami dHash (odsiewa duplikaty).
        :param max_keyframes: Maksymalna liczba zapisanych klatek kluczowych na film.
        """
        self.sample_interval_sec = sample_interval_sec
        self.min_sharpness = min_sharpness
        self.hash_diff_threshold = hash_diff_threshold
        self.max_keyframes = max_keyframes

        
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Nie udało się wczytać kaskady Haara: {cascade_path}")

    @staticmethod
    def _calculate_sharpness(gray_frame) -> float:
        return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    def _detect_faces(self, gray_frame) -> list:
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for (x, y, w, h) in faces
        ]

    def extract(self, video_path: str, output_dir: str = "keyframes_out") -> list[dict]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Nie znaleziono pliku wideo: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku wideo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        candidates = []
        saved_hashes = []
        current_sec = 0.0

        while current_sec <= duration_sec:
            # Szybkie przewijanie po czasie bez dekodowania klatek pośrednich
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = self._calculate_sharpness(gray)

            # 1. Odrzucenie klatek rozmazanych w ruchu
            if sharpness >= self.min_sharpness:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                frame_hash = imagehash.dhash(pil_img)

                # 2. Deduplikacja percepcyjna (pomijanie identycznych ujęć)
                is_duplicate = any(
                    (frame_hash - existing_hash) <= self.hash_diff_threshold
                    for existing_hash in saved_hashes
                )

                if not is_duplicate:
                    saved_hashes.append(frame_hash)
                    faces = self._detect_faces(gray)
                    candidates.append(
                        {
                            "timestamp_sec": round(current_sec, 2),
                            "sharpness": round(sharpness, 2),
                            "has_face": len(faces) > 0,
                            "faces_count": len(faces),
                            "faces_boxes": faces,
                            "dhash": str(frame_hash),
                            "frame_bgr": frame,
                        }
                    )

            current_sec += self.sample_interval_sec

        cap.release()

        # 3. Sortowanie: najpierw klatki z wykrytą twarzą, potem według najwyższej ostrości
        candidates.sort(key=lambda x: (x["has_face"], x["sharpness"]), reverse=True)
        selected = candidates[: self.max_keyframes]

        # Ponowne ułożenie chronologiczne do prezentacji w interfejsie
        selected.sort(key=lambda x: x["timestamp_sec"])

        results = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for idx, item in enumerate(selected, start=1):
            filename = f"{base_name}_kf_{idx:02d}_{item['timestamp_sec']}s.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, item["frame_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 90])

            results.append(
                {
                    "keyframe_id": idx,
                    "file_path": filepath,
                    "timestamp_sec": item["timestamp_sec"],
                    "sharpness": item["sharpness"],
                    "has_face": item["has_face"],
                    "faces_count": item["faces_count"],
                    "faces_boxes": item["faces_boxes"],
                    "dhash": item["dhash"],
                }
            )

        return results


if __name__ == "__main__":
    # Przykładowe uruchomienie testowe
    extractor = LightweightKeyframeExtractor(
        sample_interval_sec=0.5,
        min_sharpness=50.0,
        hash_diff_threshold=6,
        max_keyframes=6,
    )

    sample_video = "test_reklama.mp4"  # Podmień na ścieżkę do pobranego pliku np. z Celeb-DF
    if os.path.exists(sample_video):
        keyframes_metadata = extractor.extract(sample_video)
        print(json.dumps(keyframes_metadata, indent=2, ensure_ascii=False))
    else:
        print(f"Umieść plik '{sample_video}' w katalogu projektu, aby przetestować.")