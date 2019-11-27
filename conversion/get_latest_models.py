import csv
import sys
from collections import defaultdict


def main(argv):
    """
    :brief The main function executes the program.
    :param argv: the arguments that have been passed from the command line
    """

    reader1 = csv.reader(open(argv[0]))
    reader2 = csv.reader(open(argv[1]))
    model_name = argv[0].split('.')[0]

    architectures = []
    for row in reader1:
        architectures.append(row[0])
    architectures.sort()

    models = []
    for row in reader2:
        models.append(row[0].split('/'))
    models.sort()

    high_level_list = []
    for i in range(len(models)):
        high_level_id = int(models[i][3].replace(model_name, ''))
        if high_level_id not in high_level_list:
            high_level_list.append(high_level_id)

    level_dictionary = defaultdict(list)
    for i in range(len(models)):
        high_level_id = int(models[i][3].replace(model_name, ''))
        lower_level_id = int(models[i][4])
        if high_level_id not in level_dictionary.keys():
            level_dictionary[high_level_id].append(lower_level_id)
        elif lower_level_id not in level_dictionary[high_level_id]:
            level_dictionary[high_level_id].append(lower_level_id)

    max_iteration_models = defaultdict(list)
    for key in level_dictionary.keys():
        for i in range(len(level_dictionary[key])):
            max_iteration_models[key].append(0)

    for i in range(len(models)):
        iteration = int(models[i][5].split('.')[0].split('_')[2])
        high_level_id = int(models[i][3].replace(model_name, ''))
        lower_level_id = int(models[i][4])
        if max_iteration_models[high_level_id][lower_level_id - 1] < iteration:
            max_iteration_models[high_level_id][lower_level_id - 1] = iteration

    models = []
    for key in level_dictionary.keys():
        for i in range(len(level_dictionary[key])):
            models.append('../' + model_name + '/trained_models/' + model_name + str(key) + '/' + str(
                level_dictionary[key][i]) + '/' + 'modelsave_iter_' + str(max_iteration_models[key][i]) + '.caffemodel')
    models.sort()
    
    architectures_file = open(argv[0], "w+")
    models_file = open(argv[1], "w+")
    for i in range(len(models)):
        architectures_file.write(architectures[i] + '\n')
        models_file.write(models[i] + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
