import os
import glob


def scan_dirs_folder(folder):
    return [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def scan_files_folder(root_folder, search_exts='.*'):
    if search_exts =='.*':
        files = [x for x in glob.glob(root_folder + '\\*.*')]
        return files
    else:
        files = [x for x in glob.glob(root_folder + '\\*.*') if any(x.endswith(ext) for ext in search_exts)]
        return files


def scan_files_subfolder(root_folder, search_exts):
    searched_files = []
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if any(f.endswith(ext) for ext in search_exts):
                searched_files.append(os.path.join(root, f))
    return searched_files


def get_path_info(path):
    dir = os.path.dirname(path)
    filenmae, extension = os.path.splitext(os.path.basename(path))
    return dir, filenmae, extension