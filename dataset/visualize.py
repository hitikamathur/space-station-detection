import os
import cv2


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        classes_file = os.path.join(dataset_folder, "classes.txt")
        with open(classes_file, "r") as f:
            self.classes = f.read().splitlines()
        self.classes = {i: c for i, c in enumerate(self.classes)}
        self.set_mode(YoloVisualizer.MODE_TRAIN)

    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            self.images_folder = os.path.join(self.dataset_folder, "data", "train", "images")
            self.labels_folder = os.path.join(self.dataset_folder, "data", "train", "labels")
        else:
            self.images_folder = os.path.join(self.dataset_folder, "data", "val", "images")
            self.labels_folder = os.path.join(self.dataset_folder, "data", "val", "labels")

        self.mode = mode
        self.image_names = sorted(os.listdir(self.images_folder))
        self.label_names = sorted(os.listdir(self.labels_folder))
        self.num_images = len(self.image_names)
        assert self.num_images == len(self.label_names)
        assert self.num_images > 0
        self.frame_index = 0

    def next_frame(self):
        self.frame_index = (self.frame_index + 1) % self.num_images

    def previous_frame(self):
        self.frame_index = (self.frame_index - 1 + self.num_images) % self.num_images

    def seek_frame(self, idx):
        image_file = os.path.join(self.images_folder, self.image_names[idx])
        label_file = os.path.join(self.labels_folder, self.label_names[idx])
        image = cv2.imread(image_file)
        with open(label_file, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            class_index, x, y, w, h = map(float, line.split())
            cx = int(x * image.shape[1])
            cy = int(y * image.shape[0])
            bw = int(w * image.shape[1])
            bh = int(h * image.shape[0])
            x1 = cx - bw // 2
            y1 = cy - bh // 2
            x2 = x1 + bw
            y2 = y1 + bh
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, self.classes[int(class_index)], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image

    def run(self):
        while True:
            frame = self.seek_frame(self.frame_index)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow(f"Yolo Visualizer [{ 'TRAIN' if self.mode == self.MODE_TRAIN else 'VAL' }]", frame)
            key = cv2.waitKey(0)
            if key in [ord('q'), 27]:  # q or Esc
                break
            elif key == ord('d'):
                self.next_frame()
            elif key == ord('a'):
                self.previous_frame()
            elif key == ord('t'):
                self.set_mode(YoloVisualizer.MODE_TRAIN)
            elif key == ord('v'):
                self.set_mode(YoloVisualizer.MODE_VAL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vis = YoloVisualizer(os.path.dirname(__file__))
    vis.run()
