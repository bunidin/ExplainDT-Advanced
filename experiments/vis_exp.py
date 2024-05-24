from PIL import Image
import math


def solution_to_image(instance, sufficient_reason, image_filename):
    explanation = [(0, 0, 0) for pixel in instance]

    for i in range(len(instance)):
        if instance[i] > 1:
            explanation[i] = (120, 120, 120)
            if sufficient_reason[i] == 1:
                explanation[i] = (255, 255, 255)
        else:
            if sufficient_reason[i] == 0:
                explanation[i] = (120, 0, 0)

    size = int(math.sqrt(len(instance)))
    output = Image.new('RGB', (size, size))
    output.putdata(explanation)
    output = output.resize((size*30, size*30), resample=Image.BOX)
    output.save(image_filename)

MARGIN = 1

def flatten_row(images, margin=MARGIN, last=False):
    n_img = len(images)
    size = int(math.sqrt(len(images[0])))
    ans = []
    for row in range(size):
        for i, image in enumerate(images):
            ans += image[row*size:(row+1)*size]
            if i < n_img-1:
                ans += [(0, 0, 40) for _ in range(margin)]
    if not last:
        ans += [(0, 0, 40) for _ in range(margin*(margin*(n_img-1)+size*n_img))]
    return ans


def flatten_nine_images(nine_images):
    return flatten_row(nine_images[0:3]) + flatten_row(nine_images[3:6]) + flatten_row(nine_images[6:], last=True)


def to_rgb(instance, RFS=None):
    ans = []
    for i, el in enumerate(instance):
        in_rfs = RFS is not None and (RFS[i] in [0, 1])
        if el == 0 and not in_rfs:
            ans.append((40, 40, 40))
        if el == 0 and in_rfs:
            ans.append((0, 0, 0))
        if el and not in_rfs:
            ans.append((120, 120, 120))
        if el and in_rfs:
            ans.append((200, 200, 200))
    return ans


def RFS_to_image(instances, RFS, image_filename, v2=False):
    assert len(instances) == 8
    explanation = [(40, 40, 40) for pixel in instances[0]]

    for i in range(len(instances[0])):
        if RFS[i] == 1 or RFS[i] == 0:
            explanation[i] = (40, 140, 40)

    if True:
        if v2:
            instances = list(map(lambda instance: to_rgb(instance, RFS), instances))
        else:
            instances = list(map(to_rgb, instances))


    size = int(math.sqrt(len(instances[0])))
    margin = MARGIN
    n_img = 3
    output = Image.new('RGB', (n_img*(size+margin)-margin, n_img*(size+margin)-margin))
    nine_images = instances[0:4] + [explanation] + instances[4:]
    flat_data = flatten_nine_images(nine_images)
    output.putdata(flat_data)

    output = output.resize((size*40*3, size*40*3), resample=Image.BOX)
    output.save(image_filename)
