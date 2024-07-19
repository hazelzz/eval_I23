import os
import shutil

# Define the source and destination paths
source_folder = r"D:\wyh\eval_I23\GSO_100\GSO_100"
render_folder = r"D:\wyh\eval_I23\GSO_render_datasets"
# destination_folder = "/d:/wyh/eval_I23/render_datasets/meshes"
files = os.listdir(source_folder)
for f in files:
    print("-----------------------------",f, "-----------------------------")
    # Move the texture.png file to meshes\model.png
    shutil.copy(os.path.join(source_folder, f, r"materials\textures\texture.png"), os.path.join(source_folder, f, r"meshes\texture.png"))
    render_path = os.path.join(render_folder, f)
    os.makedirs(render_path, exist_ok=True)
    cmd = r"D:\wyh\eval_I23\blender-3.2.2-windows-x64\blender.exe --background --python blender_script.py -- --object_path {} --output_dir {} --num_images 35".format(os.path.join(source_folder, f, r"meshes\model.obj"), render_path)
    os.system(cmd)
    break

# blender-3.2.2-windows-x64\blender.exe --background --python blender_script.py -- --object_path \\tsclient\F\kaogu\ti_dataset_mesh\Ecoforms_Cup_B4_SAN\meshes\model.obj --output_dir \\tsclient\F\kaogu\ti_dataset_mesh\Ecoforms_Cup_B4_SAN  --num_images 7
# Move the files in the model folder to the subfolder itself

# source_folder = r"D:\wyh\eval_I23\GSO_render_datasets"
# files = os.listdir(source_folder)
# for f in files:
#     model_folder = os.path.join(source_folder, f, "model")
#     files_in_model = os.listdir(model_folder)
#     for file in files_in_model:
#         file_path = os.path.join(model_folder, file)
#         shutil.move(file_path, os.path.join(source_folder, f))