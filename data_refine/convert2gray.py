from PIL import Image
import os

# 定义输入和输出文件夹路径
input_folder = 'alpha_24/'
output_folder = 'alpha/'

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中所有文件的列表
file_list = os.listdir(input_folder)

# 遍历每张图片进行处理
for filename in file_list:
    # 拼接完整的文件路径
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    # 打开图片
    image = Image.open(input_path)
    
    # 转换为灰度图像
    gray_image = image.convert('L')
    
    # 保存灰度图像
    gray_image.save(output_path)

    print(f'Converted {filename} to grayscale.')

print('Batch processing complete.')
