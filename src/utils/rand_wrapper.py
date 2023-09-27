

import random
random.seed(42)

def random_from_list(input_list):
    return random.choice(input_list)


def random_sublist_from_list(input_list, number_of_elements):
    return random.choices(input_list, k=number_of_elements)


def random_from_string(input_string):
    return random.choice(input_string)


def hundred_small_random():
    list_ = [None] * 100
    for i in range(len(list_)):
        list_[i] = random.random()

    return list_

def hundred_large_random():
    list_=[None]*100
    for i in range(len(list_)):
        list_[i]=random.randrange(10,1001)
    return list_

#A fentit lehetett volna random.sample-el is csinálni, de nem fogadta el.

def five_random_number_div_three():
    list_1 = [value for value in list(range(9, 1001)) if value % 3 == 0]

    return random.sample(list_1, k = 5)

def random_reorder(input_list):
    return random.sample(input_list, k = len(input_list))



def uniform_one_to_five():
    return random.uniform(1, 6)


