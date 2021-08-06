import imageio
import os

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    orgin = 'E:/gif_image/two-gaze'
    files = os.listdir('E:/gif_image/two-gaze')
    image_list = []
    for file in files:
        path = os.path.join(orgin, file)
        image_list.append(path)
    print(image_list)
    gif_name = 'E:/gif_image/cat.gif'
    duration = 0.35
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()