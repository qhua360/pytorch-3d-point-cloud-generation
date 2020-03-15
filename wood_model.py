import numpy as np
import open3d as o3d
from PIL import Image
import options
import utils
import torch
import transform

if __name__ == "__main__":
    cfg = options.get_arguments()
    cfg.batchSize = cfg.inputViewN

    model = utils.build_structure_generator(cfg).to(cfg.device)

    print("======= IMPORT PRETRAINED MODEL =======")

    png = Image.open('data/wood/64wood.png')
    png.load()
    rgb = Image.new("RGB", png.size, (255, 255, 255))
    rgb.paste(png, mask=png.split()[3])

    image_data = np.array(rgb, dtype='uint8')

    image_data = image_data / 255.0

    arr24 = np.array([image_data])

    for i in range(23):
        arr24 = np.concatenate((arr24, np.array([image_data])))

    input_images = torch.from_numpy(arr24) \
        .permute((0, 3, 1, 2)) \
        .float().to(cfg.device)

    print("======= IMPORT IMAGE =======")

    fuseTrans = cfg.fuseTrans

    points24 = np.zeros([cfg.inputViewN, 1], dtype=np.object)

    XYZ, maskLogit = model(input_images)
    mask = (maskLogit > 0).float()
    # ------ build transformer ------
    XYZid, ML = transform.fuse3D(
        cfg, XYZ, maskLogit, fuseTrans)  # [B,3,VHW],[B,1,VHW]

    XYZid, ML = XYZid.permute([0, 2, 1]), ML.squeeze()
    for a in range(cfg.inputViewN):
        xyz = XYZid[a]  # [VHW, 3]
        ml = ML[a]  # [VHW]
        points24[a, 0] = (xyz[ml > 0]).detach().cpu().numpy()

    coords = np.zeros((0, 3))

    for l in points24[:,0]:
        coords = np.concatenate((coords, l))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.io.write_point_cloud(f"results/{cfg.model}_{cfg.experiment}/64wood.ply", pcd)

    print("======= TRANSFORM TO POINT CLOUD =======")
