import os
import megfile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def is_black_image(image):
    # 检查图像是否为完全黑色图像
    pixels = image.load()  # 获取图像像素
    for i in range(image.width):
        for j in range(image.height):
            # 如果某个像素的颜色不是黑色，则返回 False
            if pixels[i, j] != (0, 0, 0):
                return False
    return True

def split_2x2_grid(image_path, grid_size, cache_dir):
    with megfile.smart_open(image_path, 'rb') as f:
        grid_image = Image.open(f)

        width, height = grid_image.size

        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list = []

        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,      
                    i * individual_height,     
                    (j + 1) * individual_width,  
                    (i + 1) * individual_height  
                )

                individual_image = grid_image.crop(box)

                if is_black_image(individual_image):
                    print(f"Detected a black image at position ({i},{j}) in {image_path}")
                else:
                    image_list.append(individual_image)

    image_path_list = []
    for i, image in enumerate(image_list):
        image_path = os.path.join(cache_dir, f"{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)

    return image_path_list