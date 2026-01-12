import os
import tarfile
from tqdm import tqdm


def extract_and_delete_tars(base_dir):
    # 1. 获取目录下所有的 .tar 文件
    files = [f for f in os.listdir(base_dir) if f.endswith('.tar')]

    print(f"找到 {len(files)} 个待处理的 .tar 文件...")

    # 使用 tqdm 显示进度条
    for file_name in tqdm(files, desc="解压进度"):
        file_path = os.path.join(base_dir, file_name)

        # 2. 创建与文件名对应的文件夹（去掉 .tar 后缀）
        folder_name = file_name.replace('.tar', '')
        target_dir = os.path.join(base_dir, folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        try:
            # 3. 执行解压逻辑
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(path=target_dir)

            # 4. 解压成功后删除原压缩包
            os.remove(file_path)

        except Exception as e:
            print(f"\n处理文件 {file_name} 时出错: {e}")


if __name__ == "__main__":
    # 你的目标路径
    target_path = r'E:\Downloads\ILSVRC2012_img_train'

    if os.path.exists(target_path):
        extract_and_delete_tars(target_path)
        print("\n所有操作已完成！")
    else:
        print("指定的路径不存在，请检查路径是否正确。")