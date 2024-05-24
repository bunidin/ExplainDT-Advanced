from mnist import MNIST
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import dtrees.dtree as dtree
import os


# Load dataset
def setup():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(dir_path, '../datasets/mnist/')
    mndata = MNIST(dataset_path)
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return images, labels, test_images, test_labels


def binarize(image):
    result = [0 for pixel in image]
    for i in range(len(image)):
        result[i] = (image[i] > 70)*255
    return result


def binarize_images(images):
    for i, image in enumerate(images):
        images[i] = binarize(image)
    return images


def load_test_images():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(dir_path, '../datasets/mnist/')
    mndata = MNIST(dataset_path)
    images, labels = mndata.load_testing()
    return binarize_images(images), labels

# for i, image in enumerate(test_images):
    # test_images[i] = binarize(image)


def simplify_to_digit(labels, digit):
    return [int(label == digit) for label in labels]


def generate_dt(images, labels, digit, n_leaves):
    lbls = simplify_to_digit(labels, digit)
    dt = DecisionTreeClassifier(
        random_state=0,
        max_leaf_nodes=n_leaves,
        splitter='random',
        class_weight='balanced'
    )
    data_size = 60000
    train_x = binarize_images(images[:data_size])
    train_y = lbls[:data_size]
    dt.fit(train_x, train_y)
    return dt


# def get_test_images():
#     return test_images


# def get_test_labels(digit):
#     return simplify_to_digit(test_labels, digit)


def train_tree(
    images,
    labels,
    test_images,
    test_labels,
    digit,
    n_leaves,
    filename
):
    dt = generate_dt(images, labels, digit, n_leaves)
    expected = simplify_to_digit(test_labels, digit)

    predicted = dt.predict(test_images)
    acc = accuracy_score(expected, predicted)

    dimension = len(images[0])
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # filename = str(os.path.join(
    #     dir_path,
    #     f'generated_trees/mnist-{digit}-{n_leaves}.dt'
    # ))

    ft_names = [f'pixel ({i//28}, {i%28})' for i in range(dimension)]
    ft_types = ['boolean' for _ in range(dimension)]
    class_names = [f'is {digit}', f'not {digit}']

    dtree.serialize_tree_to_json(filename, dt, ft_names, ft_types, class_names, acc)
    return acc
