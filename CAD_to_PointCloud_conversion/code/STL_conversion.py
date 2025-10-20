import os
import numpy as np
import open3d as o3d

input_folder = "/Users/jiwan/Desktop/marker/CAD_to_PointCloud_conversion/dataset/input"
output_folder = "/Users/jiwan/Desktop/marker/CAD_to_PointCloud_conversion/dataset/output"
sample_count = 200000

os.makedirs(output_folder, exist_ok=True)

with os.scandir(input_folder) as it:
    for e in it:
        if not e.is_file() or not e.name.lower().endswith(".stl"):
            continue

        in_path = e.path
        out_path = os.path.join(output_folder, os.path.splitext(e.name)[0] + ".xyz")
        if os.path.exists(out_path):
            print(f"이미 처리됨: {out_path}")
            continue

        try:
            m = o3d.io.read_triangle_mesh(in_path)
            if m.is_empty():
                print(f"[건너뜀] 메시 비어있음: {in_path}")
                continue

            # 메시 정리
            m.remove_duplicated_vertices()
            m.remove_degenerate_triangles()
            m.remove_non_manifold_edges()
            if not m.has_triangle_normals():
                m.compute_triangle_normals()

            pc = m.sample_points_uniformly(number_of_points=sample_count)
            np.savetxt(out_path, np.asarray(pc.points), fmt="%.8f")
            print(f"[완료] {out_path}")
        except Exception as ex:
            print(f"[실패] {in_path}: {ex}")