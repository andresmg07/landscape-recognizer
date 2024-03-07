from os import listdir
import numpy as np

from feature_extractor import feature_vector_generator
from util import get_absolute_path, load_image, write_file, file_writing_casting, \
    read_file, file_reading_casting


class Recognizer:
    """
    General purpose image pattern recognizer.
    """
    def __init__(self, model_file_path=None):
        """
        Recognizer constructor.
        :param str model_file_path: Recognition model source file path.
        """
        self.feature_vector, self.feature_std_vector = None, None
        # If recognition model source file path provided it loads it into class attributes.
        if model_file_path:
            self.load_model(model_file_path)

    def generate_observations(self, training_set_path, observations_target_path):
        """
        Method that generates feature vector observations from a training set. Results are written into a target file.
        :param str training_set_path: Image training set source directory path.
        :param str observations_target_path: Target feature vector observations file path.
        :return: None
        """
        # Loads training set images into memory.
        training_set = [load_image('training_set/' + file_name) for file_name in
                        listdir(get_absolute_path(training_set_path))]
        # Auxiliary variable for feature vector gathering.
        model_observations = []
        for image in training_set:
            model_observations.append(feature_vector_generator(image))

        # Writes feature vector observations into target file.
        write_file(observations_target_path, file_writing_casting(model_observations))

    def generate_model(self, observations_set_path, model_path):
        """
        Method that generates a recognition model from feature vector observations. Results are written into a target file.
        :param str observations_set_path: Feature vector observations source file path.
        :param str model_path: Target recognition model file path.
        :return: None
        """
        # Loads feature vector observations into memory.
        observations = np.array(file_reading_casting(read_file(observations_set_path)))
        # Calculates feature vectors length.
        observations_count = observations.shape[0]
        # Calculates standard deviation for given observations.
        calculated_standard_deviation = np.std(observations, axis=0)
        # Calculates average feature vector with given observations.
        calculated_feature_vector = np.sum(observations, axis=0) / observations_count

        # Writes recognition model into target file.
        write_file(model_path,
                   file_writing_casting([calculated_feature_vector.tolist(), calculated_standard_deviation.tolist()]))

    def load_model(self, model_file_path):
        """
        Method that loads recognition model into class attributes.
        :param str model_file_path: Recognition model source file path.
        :return: None
        """
        # Loads recognition model into memory.
        model = file_reading_casting(read_file(model_file_path))
        # Average feature vector assignment.
        self.feature_vector = np.array(model[0])
        # Standard deviation vector assignment.
        self.feature_std_vector = np.array(model[1])

    def recognize(self, image):
        """
        Method that determines if image fit into loaded model.
        :param np.ndarray image: Targe image to compare with loaded model.
        :return: bool : Image model comparison truth value
        """
        observation_feature_vector = feature_vector_generator(image)
        upper_boundary = np.add(self.feature_vector, self.feature_std_vector)
        lower_boundary = np.subtract(self.feature_vector, self.feature_std_vector)
        truth_vector = [lower < value < upper for (value, upper, lower) in
                        zip(observation_feature_vector, upper_boundary, lower_boundary)]
        return all(truth_vector)

    def test(self, test_set_path, test_result_path, test_log_path):
        """
        Method that automates testing set result gathering process.
        :param str test_set_path: Image testing set source directory path. Results are written into several target files.
        :return: None
        """
        # Loads test file names into memory
        test_file_names = listdir(get_absolute_path(test_set_path))
        # Loads test files into memory
        testing_set = [load_image('testing_set/' + file_name) for file_name in test_file_names]
        # Initialization of counters
        true_negative = 0
        true_positive = 0
        false_negative = 0
        false_positive = 0
        # Initialization of testing log string.
        test_log = ''
        for file_name, image in zip(test_file_names, testing_set):
            # Image fitness evaluation.
            does_image_fit_model = self.recognize(image)
            # Append fitness result into log string.
            test_log += f'''{file_name} - {does_image_fit_model}'''
            # Result type characterization
            if does_image_fit_model and file_name[0] == 't':
                true_positive += 1
                test_log += ' - tp\n'
            if does_image_fit_model and file_name[0] == 'f':
                false_positive += 1
                test_log += ' - fp\n'
            if not does_image_fit_model and file_name[0] == 'f':
                true_negative += 1
                test_log += ' - tn\n'
            if not does_image_fit_model and file_name[0] == 't':
                false_negative += 1
                test_log += ' - fn\n'

        # Writes test results into target file.
        write_file(test_result_path, f'''{true_negative}\n{true_positive}\n{false_negative}\n{false_positive}''')
        # Writes test logs into target file.
        write_file(test_log_path, test_log)

