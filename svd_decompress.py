import numpy as np
from PIL import Image

def svd_to_matrix(U_k,sv_k,VT_k):
    A = U_k @ np.diag(sv_k) @ VT_k
    return A

def matrices_to_img(R,G,B):
    img_mat = np.zeros((R.shape[0],R.shape[1],3))
    img_mat[:,:,0] = R
    img_mat[:,:,1] = G
    img_mat[:,:,2] = B
    img = Image.fromarray(np.clip(img_mat, 0, 255).astype(np.uint8))
    return img

def remove_file_type(path):
    dot_index = path.rfind(".")
    if dot_index == -1:
        name = path
    else:
        name = path[:dot_index]
    return name

name = input("Enter the file name to decompress:  ")
data = np.load(f"npzs/{name}")
rank = data["rank"]
R_U_k = data["R_U_k"]
R_sv_k = data["R_sv_k"]
R_VT_k = data["R_VT_k"]

G_U_k = data["G_U_k"]
G_sv_k = data["G_sv_k"]
G_VT_k = data["G_VT_k"]

B_U_k = data["B_U_k"]
B_sv_k = data["B_sv_k"]
B_VT_k = data["B_VT_k"]

R = svd_to_matrix(R_U_k, R_sv_k, R_VT_k)
G = svd_to_matrix(G_U_k, G_sv_k, G_VT_k)
B = svd_to_matrix(B_U_k, B_sv_k, B_VT_k)

name = remove_file_type(name)

img = matrices_to_img(R, G, B)
img.save(f"decompressed_images/{name}.png")

print(f"Decompressed file saved as decompressed_images/{name}.png")




