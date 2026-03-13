import cv2  # Important !!! pip install opencv-contrib-python
import numpy as np

from scipy.ndimage import uniform_filter1d


class LineDetector:
    def __init__(self):
        pass

    def transform(self, frame):
        pts1 = np.float32([[0, 260], [640, 260],
                           [0, 400], [640, 400]])
        pts2 = np.float32([[0, 0], [400, 0],
                           [0, 640], [400, 640]])

        # Apply Perspective Transform Algorithm
        try:
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(frame, matrix, (500, 600))
        except:
            raise "No Lines Detected"

        return result

    # getting binary mask for the lines
    def threshold_img(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Making image in hsv formate

        lower_bright = np.array([0, 100, 100])
        upper_bright = np.array([179, 255, 255])

        mask = cv2.inRange(hsv, lower_bright, upper_bright)  # getting binary mask

        return mask

    def middle_point(self, line_clusters):
        # mean 1 line
        mean_x0 = float(np.mean(line_clusters[0][:, 0]))
        mean_y0 = float(np.mean(line_clusters[0][:, 1]))
        # mean 2 line
        mean_x1 = float(np.mean(line_clusters[1][:, 0]))
        mean_y1 = float(np.mean(line_clusters[1][:, 1]))

        # middle point
        mid_x = (mean_x0 + mean_x1) / 2.0
        mid_y = (mean_y0 + mean_y1) / 2.0

        return mid_x, mid_y

    def skeletonization_img(self, black_and_white_mask):
        skeleton = cv2.ximgproc.thinning(black_and_white_mask)  # using skeleton function

        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), connectivity=8)

        areas = stats2[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]  # sorting down
        top = order[:2] + 1  # we exclude background

        line_clusters = []
        for lab in top:
            ys, xs = np.where(labels2 == lab)
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)  # optimized array
            line_clusters.append(pts)

        return line_clusters

    def optimize_frame(self, frame):
        # resizing the image
        resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        resized_blured = cv2.medianBlur(resized, 25)  # cv2.blur(resized, (9, 9))
        resized_blured = cv2.GaussianBlur(resized_blured, (11, 11), 0)
        return resized_blured

    def Morphology(self, black_and_white_mask):
        # Morphology - connects fragmented lines if exist
        kernel = np.ones((5, 5), np.uint8)
        black_and_white_mask = cv2.morphologyEx(black_and_white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        black_and_white_mask = cv2.morphologyEx(black_and_white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return black_and_white_mask

    def midle_line(self, line_clusters):
        # getting lines
        line1 = line_clusters[0][np.argsort(line_clusters[0][:, 1])]
        line2 = line_clusters[1][np.argsort(line_clusters[1][:, 1])]

        line1_dic = {}
        line2_dic = {}

        # getting rid of duplicates
        for x, y in line1:
            if y not in line1_dic:
                line1_dic[y] = []
            line1_dic[y].append(x)

        for x, y in line2:
            if y not in line2_dic:
                line2_dic[y] = []
            line2_dic[y].append(x)

        common_ys = sorted(set(line1_dic.keys()) & set(line2_dic.keys()))  # only with common y
        mid_points = []

        for y in common_ys[::4]:
            x1 = int(np.mean(line1_dic[y]))
            x2 = int(np.mean(line2_dic[y]))

            mid_x = int((x1 + x2) / 2)
            mid_points.append((mid_x, y))

        xs = np.array([p[0] for p in mid_points])
        ys = np.array([p[1] for p in mid_points])

        xs_s = uniform_filter1d(xs, size=7)
        ys_s = uniform_filter1d(ys, size=7)

        smoothed_data = list(zip(xs_s.astype(int), ys_s.astype(int)))

        return smoothed_data, mid_points

    def process_frame(self, frame):
        try:
            optimized_frame = self.optimize_frame(frame)  # frame optimization
            transformed_frame = self.transform(optimized_frame)
            black_and_white_mask = self.threshold_img(transformed_frame)  # binary mask
            resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            transformed_frame_orig = self.transform(frame)
            resized_frame = transformed_frame_orig
            # morphology
            morphology_mask = self.Morphology(black_and_white_mask)
            # detection
            line_clusters = self.skeletonization_img(morphology_mask)

            colors = [(255, 0, 0), (0, 255, 0)]  # BGR
            if len(line_clusters) >= 2:
                for i, cluster in enumerate(line_clusters):
                    for x, y in cluster:
                        cv2.circle(resized_frame, (int(x), int(y)), 2, colors[i], -1)

                if len(line_clusters) > 0:
                    # middle line
                    mid_x, mid_y = self.middle_point(line_clusters)
                    smoothed_data, mid_points = self.midle_line(line_clusters)

                    for i in range(len(mid_points) - 1):
                        cv2.line(resized_frame, smoothed_data[i], smoothed_data[i + 1], (0, 0, 255), 2)

                    # graphing results
                    cv2.circle(resized_frame, (int(mid_x), int(mid_y)), 8, (255, 255, 255), -1)  # dot
        except:
            raise "Error"
        return resized_frame