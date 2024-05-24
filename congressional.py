from sklearn.tree import DecisionTreeClassifier
import dtrees.dtree as dtree
import csv


dataset_filename = 'datasets/house-votes-84.data'
full_dataset = None
with open(dataset_filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    full_dataset = list(reader)

non_missing_values = []
labels = []

def to_binary(record):
    return [1 if value == 'y' else 0 for value in record]


for record in full_dataset:
    if '?' not in record:
        non_missing_values.append(to_binary(record[1:]))
        labels.append(record[0])


class_names = ['democrat', 'republican']


def process_labels(labels):
    ans = []
    for label in labels:
        ans.append(class_names.index(label))
    return ans


labels = process_labels(labels)

X, y = non_missing_values, labels

congressional_classifier = DecisionTreeClassifier(max_leaf_nodes=400,
                                                  random_state=0)
congressional_classifier.fit(X, y)

print('DecisionTreeClassifier has been trained!')


features = [
        ('handicappedInfants', 'boolean'),
        ('waterCostSharing', 'boolean'),
        ('adoptionBudgetRes', 'boolean'),
        ('physicianFeeFreeze', 'boolean'),
        ('aidElSalvador', 'boolean'),
        ('religiousGroupsSchools', 'boolean'),
        ('antiSatelliteTestBan', 'boolean'),
        ('aidNicaraguan', 'boolean'),
        ('supportsMxMissile', 'boolean'),
        ('supportsImmigration', 'boolean'),
        ('synFuelsCutback', 'boolean'),
        ('educationSpending', 'boolean'),
        ('superfundRightSue', 'boolean'),
        ('crime', 'boolean'),
        ('dutyFreeExports', 'boolean'),
        ('expAdminSouthAfrica', 'boolean'),
        ]

ft_names = list(map(lambda x: x[0], features))
ft_types = list(map(lambda x: x[1], features))

export_filename = 'congressional_dt.json'

dtree.serialize_tree_to_json(export_filename, congressional_classifier,
                             ft_names, ft_types, class_names)

print(f'Exported to: {export_filename}')
