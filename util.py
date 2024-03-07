import os
import platform
import cv2


def write_file(relative_file_path, content):
    """
    Utilitarian procedure that writes text content into .txt file.
    :param str relative_file_path: Relative file path from the project root directory.
    :param str content: Text content to be written into target file.
    :return: None
    """
    try:
        file_path = get_absolute_path(relative_file_path)
        file = open(file_path, "w")
        file.write(content)
        file.close()
    except Exception:
        raise Exception(f'''File writing unsuccessful on {relative_file_path}.''')


def read_file(relative_file_path):
    """
    Utilitarian function that reads content from a file.
    :param str relative_file_path: Relative file path from the project root directory.
    :return: str : File content
    """
    try:
        file_path = get_absolute_path(relative_file_path)
        file = open(file_path, "r")
        content = file.read()
        file.close()
        return content
    except Exception:
        raise Exception(f'''File reading unsuccessful on {system_dependant_path_formatter(relative_file_path)}.''')


def load_image(relative_file_path):
    """
    Utilitarian function that loads into memory an image file with OpenCV library.
    :param str relative_file_path: Relative file path from the project root directory.
    :return: ndarray: Target image loaded.
    """
    file_path = get_absolute_path(relative_file_path)
    if os.path.isfile(file_path):
        return cv2.imread(file_path)
    else:
        raise Exception(f'''Image open unsuccessful with path {file_path}.''')


def get_absolute_path(relative_path):
    """
    Utilitarian function that returns the absolute path from a relative path inside the project.
    :param str relative_path: Target relative path to be converted into absolute path.
    :return: str : Absolute path
    """
    separator = system_dependant_path_formatter('/')
    return system_dependant_path_formatter('/'.join(__file__.split(separator)[:-1]) + '/' + relative_path)


def file_writing_casting(data_set):
    """
    Utilitarian function that casts a nested list into type and format for file writing.
    :param data_set: Nested list structure to be cast.
    :return: str : Flattened data set.
    """
    casted_features = [(str(feature) for feature in data_point) for data_point in data_set]
    return '\n'.join(",".join(map(str, x)) for x in casted_features)


def file_reading_casting(file_content):
    """
    Utilitarian function that cast file content into nested list.
    :param file_content: Plain text to be cast.
    :return: list of (list of float) : Structured file content.
    """
    return [list(map(float, data_point.split(','))) for data_point in file_content.split('\n')]


def show_image(image):
    """
    Utilitarian procedure that shows an image on screen with OpenCV library.
    :param ndarray image: Target image to be shown.
    :return: None
    """
    cv2.imshow("images", image)
    cv2.waitKey(0)


def system_dependant_path_formatter(path):
    """
    Utilitarian function that formats path separator character depending on operating system.
    :param str path: Path to be formatted.
    :return: str : OS dependant formatted path.
    """
    if platform.system() == 'Windows':
        return path.replace('/', '\\')
    else:
        return path.replace('\\', '/')


def get_image_area(image):
    """
    Utilitarian function that calculates the area (height x width) in pixels of an image.
    :param ndarray image: Target image to calculate its area.
    :return: int: Image area.
    """
    height, width = image.shape
    return height * width
