#2026/4/5
#Compress the image in image/
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def upload_img():
    path = input("Enter the file name of the image:  ")
    img = Image.open(f"images/{path}")
    return img,path

def img_to_matrices(img):
    img_mat = np.array(img)
    R = img_mat[:,:,0]
    G = img_mat[:,:,1]
    B = img_mat[:,:,2]
    return R, G, B

def svd_compress(A,rank):
    U,sv,VT = np.linalg.svd(A,full_matrices=False)
    U_k = U[:,:rank]
    sv_k = sv[:rank]
    VT_k = VT[:rank,:]
    A_k = U_k @ np.diag(sv_k) @ VT_k
    return A_k, U_k, sv_k, VT_k

def create_sv_loss_plot(A):
    U,sv,VT = np.linalg.svd(A,full_matrices=False)
    total = np.sum(sv**2)
    portion = sv**2/total
    rank = np.arange(1,len(sv)+1)
    loss = 1 - np.cumsum(portion)
    return rank,loss

def remove_file_type(path):
    dot_index = path.rfind(".")
    if dot_index == -1:
        name = path
    else:
        name = path[:dot_index]
    return name

img,path = upload_img()
R, G, B = img_to_matrices(img)
R_rank, R_loss = create_sv_loss_plot(R)
G_rank, G_loss = create_sv_loss_plot(G)
B_rank, B_loss = create_sv_loss_plot(B)
plt.plot(R_rank,R_loss,color="red",label="SV of Red Channel")
plt.plot(G_rank,G_loss,color="green",label="SV of Green Channel")
plt.plot(B_rank,B_loss,color="blue",label="SV of Blue Channel")
    
plt.xlabel("Rank")
plt.ylabel("Data loss")
plt.title("Rank-Data Loss Plot")
plt.show()
rank = int(input("Enter the compression rank:  "))
R_A_k,R_U_k,R_sv_k,R_VT_k = svd_compress(R,rank)
G_A_k,G_U_k,G_sv_k,G_VT_k = svd_compress(G,rank)
B_A_k,B_U_k,B_sv_k,B_VT_k = svd_compress(B,rank)

name = remove_file_type(path)

np.savez(f"npzs/{name}_rank{rank}.npz",R_U_k = R_U_k, R_sv_k = R_sv_k, R_VT_k = R_VT_k,
                            G_U_k = G_U_k, G_sv_k = G_sv_k, G_VT_k = G_VT_k,
                            B_U_k = B_U_k, B_sv_k = B_sv_k, B_VT_k = B_VT_k, rank = rank)

print(f"Compressed file saved as npzs/{name}_rank{rank}.npz")