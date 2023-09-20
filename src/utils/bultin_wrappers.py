# 1., Hozzon létre egy listát a következő elemekkel a Subscript operátor segítségével: 17, 18, 3.14, a, alma
list1=[17, 18, 3.14, "a", "alma"]


# 7., Írjon egy függvényt ami megvizsgálja, hogy a listában létezik-e egy adott elem
# függvény név: contains_value
# bemeneti paraméterek: input_list, element
# kimeneti típus: bool

def contains_value(input_list, element):
    if element in input_list:
        return True
    else:
        return False


# 8., Írjon egy függvényt ami megvizsgálja, hogy hány elem található a listában.
# függvény név: number_of_elements_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: int

def number_of_elements_in_list(input_list):
    count = 0
    for element in input_list:
        count += 1
    return count

# 9., Írjon egy függvényt ami törli az összes elemet a listából
# függvény név: remove_every_element_from_list
# bemeneti paraméterek: input_list
# kimeneti típus: None

def remove_every_element_from_list(input_list):
    for i in range(len(input_list)):
        del input_list[0]
    return input_list


def reverse_list(input_list):
    x = [None]*len(input_list)
    for i in range(len(input_list)):
        x[i]=input_list[(len(input_list)-i-1)]
    input_list=x
    return input_list





def odds_from_list(input_list):
    return [i for i in range(len(input_list)) if i % 2 == 1]




def number_of_odds_in_list(input_list):
    count=0
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            count += 1
    return count


def contains_odd(input_list):
    count = 0
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            count += 1
    if count == 0:
        return False
    else:
        return True


def second_largest_in_list(input_list):
    input_list.sort()
    return input_list[len(input_list) - 2]


def sum_of_elements_in_list(input_list):
    sum_el=0
    for i in range(len(input_list)):
        sum_el += input_list[i]
    return float(sum_el)


def cumsum_list(input_list):
    cumsum = [None] * len(input_list)
    for i in range(len(input_list)):
        input_list_mod = input_list[0:i + 1]
        sum_el = 0
        for j in range(i + 1):
            sum_el += input_list_mod[j]
        cumsum[i] = sum_el
    return cumsum


def element_wise_sum(input_list1, input_list2):
    x = [None] * len(input_list1)
    for i in range(len(input_list1)):
        x[i] = input_list1[i] + input_list2[i]

    return x



def subset_of_list(input_list, start_index, end_index):
    x = input_list[start_index:end_index+1]
    return x


def every_nth(input_list, step_size):
    x = [None] * (len(input_list) // step_size + 1)
    for i in range(len(x)):
        x[i] = input_list[0:(len(input_list)):step_size][i]
    return x


def only_unique_in_list(input_list):
    sum_el = 0
    for i in range(len(input_list)):
        dummylist = input_list[0:i] + input_list[i + 1:(len(input_list))]
        if input_list[i] in dummylist:
            sum_el += 1
        else:
            sum_el += 0
    if sum_el == 0:
        return True
    else:
        return False


def keep_unique(input_list):
    for i in range(len(input_list)):
        reverse_index = len(
            input_list) - i - 1  # Azért reverse, hogy a későbbi duplikátumo(ka)t távolítsa el, ne az előbbi(eke)t
        dummylist = input_list[0:reverse_index] + input_list[reverse_index + 1:(len(input_list))]
        if input_list[reverse_index] in dummylist:
            input_list[reverse_index] = 'duplicate'

    # if input_list[reverse_index]=="duplicate":
    #     del input_list[reverse_index]

    input_list[:] = (value for value in input_list if value != "duplicate")

    return input_list


def swap(input_list, first_index, second_index):
    first = input_list[first_index]
    last = input_list[second_index]
    input_list[first_index] = last
    input_list[second_index] = first

    return input_list


def remove_element_by_value(input_list, value_to_remove):
    for i in range(len(input_list)):
        if input_list[i] == value_to_remove:
            input_list[i] = "Needs to be removed"

    input_list[:] = (value for value in input_list if value != "Needs to be removed")

    return input_list


def remove_element_by_index(input_list, index):
    for i in range(len(input_list)):
        if i == index:
            input_list[i] = "Needs to be removed"

    input_list[:] = (value for value in input_list if value != "Needs to be removed")

    return input_list


def multiply_every_element(input_list, multiplier):
    for i in range(len(input_list)):
        input_list[i] = multiplier * input_list[i]

    return input_list


def remove_key(input_dict, key):
    if key in input_dict.keys():
        del input_dict[key]
        return input_dict
    else:
        return input_dict


def sort_by_key(input_dict):
    return dict(sorted(input_dict.items()))



def merge_two_dicts(input_dict1, input_dict2):
    conc_dict = input_dict1 | input_dict2
    return conc_dict



def merge_dicts(*dicts):
    superdict=dict()
    for i in range(len(dicts)):
        superdict.update(dicts[i])
    return superdict



def sort_list_by_parity(input_list):
    even=list()
    odd=list()
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            odd.append(input_list[i])
        else:
            even.append(input_list[i])
    superdict= {
  "even": even,
  "odd": odd
}
    return superdict


def mean_by_key_value(input_dict):
    dummydict = dict()
    for i in input_dict:
        if len(input_dict[i]) < 2:
            dummydict[i] = input_dict[i][0]
        else:
            dummydict[i] = sum(input_dict[i]) / len(input_dict[i])

    return dummydict


def count_frequency(input_list):
    count = [None] * len(input_list)
    label = [None] * len(input_list)
    for i in range(len(input_list)):
        count[i] = input_list.count(input_list[i])
        label[i] = input_list[i]
        dummy = label[0:i] + label[i + 1:len(input_list)]
        if label[i] in dummy:
            label[i] = "duplicate"
            count[i] = "duplicate"

    label[:] = (value for value in label if value != "duplicate")
    count[:] = (value for value in count if value != "duplicate")

    return dict(zip(label, count))


