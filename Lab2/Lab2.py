import csv
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_xyz(filename):
    points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if len(row) >= 3:
                try:
                    x, y, z = float(row[0]), float(row[1]), float(row[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
    return np.array(points)


def plot_clusters(points, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    for label in unique_labels:
        cluster = points[labels == label]
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                   s=1, c=[cmap(label)])
    ax.set_title(title)
    plt.show()


def average_distance_to_plane(points, model):
    normal, d = model
    distances = np.abs(np.dot(points, normal) + d)
    return np.mean(distances)


def classify_plane(normal, horizontal_threshold=0.8):
    if abs(normal[2]) >= horizontal_threshold:
        return "pozioma"
    else:
        return "pionowa"

def ransac_plane_fit(points, max_iterations=200, distance_threshold=0.1, min_inliers=100):
    best_inliers = []
    best_model = None
    n_points = points.shape[0]

    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-6:
            continue

        normal = normal / norm_normal
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(points, normal) + d)
        inlier_idxs = np.where(distances < distance_threshold)[0]

        if len(inlier_idxs) > len(best_inliers):
            best_inliers = inlier_idxs
            best_model = (normal, d)

    if best_model is not None and len(best_inliers) >= 3:
        inlier_pts = points[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid)
        refined_normal = vh[-1, :]
        refined_d = -np.dot(refined_normal, centroid)
        best_model = (refined_normal, refined_d)

    return best_model, best_inliers


def main():
    filename = "../Lab2/all_surfaces.xyz"
    points = read_xyz(filename)
    print(f"Liczba punktów w chmurze: {points.shape[0]}")

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(points)
    plot_clusters(points, labels_kmeans, "Klasteryzacja K-średnie (k=3)")

    print("Wyniki dopasowania RANSAC (własna implementacja):")
    for cluster_id in np.unique(labels_kmeans):
        cluster_points = points[labels_kmeans == cluster_id]
        print(f"\nChmura punktów klastru {cluster_id}: liczba punktów = {cluster_points.shape[0]}")
        model, inlier_idxs = ransac_plane_fit(cluster_points,
                                              max_iterations=200,
                                              distance_threshold=0.1,
                                              min_inliers=100)
        if model is None:
            print("Nie znaleziono modelu płaszczyzny.")
            continue

        normal, d = model
        print(f"Współrzędne wektora normalnego: {normal}")
        avg_dist = average_distance_to_plane(cluster_points, model)
        print(f"Średnia odległość punktów od płaszczyzny: {avg_dist:.4f}")

        if avg_dist < 0.1:
            print("Chmura ta jest płaszczyzną.")
            orientation = classify_plane(normal, horizontal_threshold=0.8)
            print(f"Orientacja płaszczyzny: {orientation}")
        else:
            print("Chmura ta nie jest płaszczyzną.")

    #DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=10)
    labels_dbscan = dbscan.fit_predict(points)
    plot_clusters(points, labels_dbscan, "Klasteryzacja DBSCAN (dostrojona)")

    n_clusters = len(np.unique(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"Liczba klastrów DBSCAN: {n_clusters}")

    try:
        from pyransac3d import Plane
    except ImportError:
        print("Brak pakietu pyransac3d. Zainstaluj go przez 'pip install pyransac3d'")
        return

    print("\nWyniki dopasowania RANSAC (pyransac3d) dla DBSCAN:")
    for label in np.unique(labels_dbscan):
        if label == -1:
            continue
        cluster_points = points[labels_dbscan == label]
        print(f"\nKlaster {label}: liczba punktów = {cluster_points.shape[0]}")

        plane_model = Plane()
        best_eq, inliers = plane_model.fit(cluster_points, thresh=0.001, maxIteration=1000)
        normal = np.array(best_eq[:3])
        d = best_eq[3]

        avg_dist = average_distance_to_plane(cluster_points, (normal, d))
        print(f"Średnia odległość: {avg_dist:.4f}")

        if avg_dist < 0.1:
            orientation = classify_plane(normal)
            print(f"Znaleziono płaszczyznę {orientation}")
        else:
            print("To nie jest płaszczyzna")

if __name__ == "__main__":
    main()
