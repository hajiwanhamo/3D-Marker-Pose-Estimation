import os, numpy as np, open3d as o3d

# 조절 파라미터
POINT_SIZE = 5.0      
SEPARATION = 0.0    

def visualize_xyz(file_path):
    try:
        points = np.loadtxt(file_path)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if SEPARATION:  # 분리(간격) 조절
            pc = pc.voxel_down_sample(float(SEPARATION))

        vis = o3d.visualization.Visualizer()
        vis.create_window(f"시각화: {os.path.basename(file_path)}")
        vis.add_geometry(pc)
        vis.get_render_option().point_size = float(POINT_SIZE)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"[오류] {file_path} 시각화 실패: {e}")

# output_folder = "/Users/jiwan/Desktop/marker/CAD_to_PointCloud_conversion/dataset/output" # CAD파일 변환 이후 만들어진 xyz파일 시각화용
output_folder = "/Users/jiwan/Desktop/marker/CAD_to_PointCloud_conversion/dataset/augmentation/xyz" # 기존 마커 파일을 증강하고 증강된 파일들을 시각화 하는 용도

for file_name in os.listdir(output_folder):
    if file_name.lower().endswith(".xyz"):
        visualize_xyz(os.path.join(output_folder, file_name))
