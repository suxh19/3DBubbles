import stltovoxel
import os

output_pathdir = 'combined'
os.makedirs(output_pathdir, exist_ok=True)

stltovoxel.convert_file('3Dbubbleflowrender/20250926T112212-646384-flow0000/0.01.stl', output_pathdir+'/1.png', parallel=True, resolution = 500)