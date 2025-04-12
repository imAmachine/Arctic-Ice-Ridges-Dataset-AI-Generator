import cv2
import numpy as np


class RotationAnalyze:
    @staticmethod
    def get_max_abs_angle(angles: dict):
        return int(max(map(lambda x: abs(x), list(angles.values()))))
    
    @staticmethod
    def get_weighted_angle(angles: dict):
        weights = {
            'pca': 0.6,
            'rect': 0.2,
            'hough': 0.2
        }
        
        weighted_angle = 0
        
        for method, angle in angles.items():
            weight = weights.get(method)
            if weight:
                weighted_angle += angle * weight

        return weighted_angle
    
    @staticmethod
    def get_consistent_angle(angles: dict[str: float]):        
        return np.median(list(angles.values()))
    
    @staticmethod
    def get_PCA_rotation_angle(image: np.ndarray):
        y, x = np.where(image > 0)
        if len(x) < 2:
            return 0.0
        
        data = np.column_stack((x.astype(np.float32), y.astype(np.float32)))
        cov_matrix = np.cov(data - np.mean(data, axis=0), rowvar=False)
        _, eigenvectors = np.linalg.eig(cov_matrix)
        return np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    @staticmethod
    def get_rect_rotation_angle(image: np.ndarray):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        
        width, height = rect[1]
        if width < height:
            angle += 90
        
        return angle
    
    @staticmethod
    def get_hough_rotation_angle(image: np.ndarray):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
            
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            angles.append(angle_deg)
            
        from sklearn.cluster import KMeans
        if len(angles) > 5:
            kmeans = KMeans(n_clusters=3).fit(np.array(angles).reshape(-1, 1))
            counts = np.bincount(kmeans.labels_)
            dominant_cluster = np.argmax(counts)
            dominant_angles = [angles[i] for i in range(len(angles)) if kmeans.labels_[i] == dominant_cluster]
            return np.median(dominant_angles)
        else:
            return np.median(angles)