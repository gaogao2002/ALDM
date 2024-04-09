from PIL import Image
def get_real_images_path(data_path,pairs):
    path = "/home/bh/gaobo/try_on/result/real/"
    for i in range(len(pairs)):
        img = Image.open(data_path +"/test/image/"+pairs[i][1])
        img.save(path+str(i)+".png")

data_path = "/home/bh/gaobo/try_on/dataset/VITON-HD"
data_pair = list(open("/home/bh/gaobo/try_on/dataset/VITON-HD/test_pairs.txt"))
for i in range(len(data_pair)):
    flag = data_pair[i].find("jpg")
    path1 = data_pair[i][0:flag+3]
    path2 = data_pair[i][flag+4:-1]
    data_pair[i] = [path1,path2]

get_real_images_path(data_path,data_pair)