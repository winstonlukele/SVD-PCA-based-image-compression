from PIL import Image
import numpy as np

def upload_img():
    path = input("Enter the file name of the image.")
    img = Image.open(f"images/{path}")
    return img

def img_to_matrices(img):
    img_mat = np.array(img)
    R = img_mat[:,:,0]
    G = img_mat[:,:,1]
    B = img_mat[:,:,2]
    return R, G, B

def print_sv_info(A):
    U,sv,VT = np.linalg.svd(A,full_matrices=False)
    total = np.sum(sv**2)
    for i in range(min(len(sv),50)):
        lambda_i = sv[i]**2
        print(f"Principal Component: {i}, data captured: {lambda_i/total*100:.5f}%")

def svd_compress(A,rank):
    U,sv,VT = np.linalg.svd(A,full_matrices=False)
    U_k = U[:,:rank]
    sv_k = sv[:rank]
    VT_k = VT[:rank,:]
    A_k = U_k @ np.diag(sv_k) @ VT_k
    return A_k, U_k, sv_k, VT_k

def matrices_to_img(R,G,B):
    img_mat = np.zeros((R.shape[0],R.shape[1],3))
    img_mat[:,:,0] = R
    img_mat[:,:,1] = G
    img_mat[:,:,2] = B
    img = Image.fromarray(np.clip(img_mat, 0, 255).astype(np.uint8))
    return img

def save_img_approx(img):
    pass

def create_rgb_sv_plot(R,G,B): #Create and show a plot of R,G,B sv portions.

    return plot



# Get image uploaded
img = upload_img()

#For test only, show the image
#img.show()

# Decompose the image to its RGB channels
red_channel, green_channel, blue_channel = img_to_matrices(img)

# Get the portions of data of principal components
print("SV info of red channel")
print_sv_info(red_channel)
print("SV info of green channel")
print_sv_info(green_channel)
print("SV info of blue channel")
print_sv_info(blue_channel)
#Show the portion plot
#TODO
#SVD compress
rank = int(input("Enter the compression rank"))
R_approx = svd_compress(red_channel,rank)[0]
G_approx = svd_compress(green_channel,rank)[0]
B_approx = svd_compress(blue_channel,rank)[0]
#Add three channels to reach the result image, saved as output/image.jpg
img_approx = matrices_to_img(R_approx,G_approx,B_approx)

#Save and show the compressed image

img_approx.show()