from PIL import Image
import os
"""
    将一张GIF动图分解到指定文件夹
    src_path：要分解的gif的路径
    dest_path：保存后的gif路径
"""
def gifSplit(src_path, dest_path, suffix="png"):
    img = Image.open(src_path)
    for i in range(img.n_frames):
        img.seek(i)
        new = Image.new("RGBA", img.size)
        new.paste(img)
        new.save(os.path.join(dest_path, "%d.%s" %(i, suffix)))
 
 
 
path = 'E:/gif_image/one/two.gif'
gifSplit(path, r'E:/gif_image/two/')
