import os
import hydra
from os.path import expanduser, join

def find_all_files(directory, extension='.flac'):
    """
    在指定目录及其子目录中查找所有以指定扩展名结尾的文件。
    返回包含文件路径的列表。
    
    参数:
        directory (str): 要搜索的目录。
        extension (str): 文件扩展名（默认为 '.flac'）。
    
    返回:
        list: 包含绝对路径的文件列表。
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def write_filelist(file_list, output_file):
    """
    将文件路径列表写入指定的文件中，每行一个路径。
    
    参数:
        file_list (list): 文件路径列表。
        output_file (str): 输出文件的路径。
    """
    with open(output_file, 'w') as f:
        for file_path in file_list:
            f.write(file_path + '\n')

@hydra.main(version_base=None, config_path='config', config_name='default')
def preprocess(cfg):
    os.makedirs('filelists', exist_ok=True)
    
    root = expanduser(cfg.preprocess.datasets.LibriSpeech.root)
    
    # 处理训练集
    trainfiles = []
    print(f'Root: {root}')
    for subset in cfg.preprocess.datasets.LibriSpeech.trainsets:
        subset_path = join(root, subset)
        files = find_all_files(subset_path, '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        trainfiles.extend(files)
    
    output_train = cfg.preprocess.view.train_filelist
    print(f'Writing train filelist to {output_train}')
    write_filelist(trainfiles, output_train)
    
    # 处理测试集
    testfiles = []
    print(f'Root: {root}')
    for subset in cfg.preprocess.datasets.LibriSpeech.testsets:
        subset_path = join(root, subset)
        files = find_all_files(subset_path, '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        testfiles.extend(files)
    
    output_test = cfg.preprocess.view.test_filelist
    print(f'Writing test filelist to {output_test}')
    write_filelist(testfiles, output_test)
    
    # 处理验证集
    valfiles = []
    print(f'Root: {root}')
    for subset in cfg.preprocess.datasets.LibriSpeech.valsets:
        subset_path = join(root, subset)
        files = find_all_files(subset_path, '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        valfiles.extend(files)
    
    output_val = cfg.preprocess.view.val_filelist
    print(f'Writing val filelist to {output_val}')
    write_filelist(valfiles, output_val)

if __name__ == '__main__':
    preprocess()