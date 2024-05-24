import random

def random_tall_rectangle(SIDE):
    width = random.randint(1, SIDE-1)
    height = random.randint(width+1, SIDE)
    left = random.randint(0, SIDE - width)
    bottom = random.randint(0, SIDE - height)
    return ((left, bottom), (left+width-1, bottom+height-1))

def random_wide_rectangle(SIDE):
    height = random.randint(1, SIDE-1)
    width = random.randint(height+1, SIDE)
    left = random.randint(0, SIDE - width)
    bottom = random.randint(0, SIDE  - height)
    return ((left, bottom), (left+width-1, bottom+height-1))

def rec_to_image(rec, SIDE):
    ans = [0 for i in range(SIDE*SIDE)]
    ((left, bottom), (right, top)) = rec
    for i in range(left, right+1):
       ans[(SIDE-1-top)*SIDE + i] = 255
       ans[(SIDE-1-bottom)*SIDE + i] = 255
    for j in range(bottom, top+1):
        ans[(SIDE-1-j)*SIDE + left] = 255
        ans[(SIDE-1-j)*SIDE + right] = 255
    return ans

def random_tall_image(SIDE):
    return rec_to_image(random_tall_rectangle(SIDE), SIDE)

def random_wide_image(SIDE):
    return rec_to_image(random_wide_rectangle(SIDE), SIDE)

def random_image(TARGET_CLASS, SIDE):
    if TARGET_CLASS == 'wide':
        return random_wide_image(SIDE)
    else:
        return random_tall_image(SIDE)

def get_random_dataset(n, SIDE):
    tall_set = set()
    wide_set = set()
    for i in range(n):
        tall_set.add(random_tall_rectangle(SIDE))
        wide_set.add(random_wide_rectangle(SIDE))
    def rti(rec):
        return rec_to_image(rec, SIDE)
    data = list(map(rti, list(tall_set))) + list(map(rti, list(wide_set)))
    labels = ['tall' for i in range(len(tall_set))] + ['wide' for i in range(len(wide_set))]
    return data, labels

