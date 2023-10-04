#Neptun-kód: ZALLWZ


def evens_from_list(input_list):
    dummy = input_list.copy()
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            dummy[i] = "Not even"

    dummy = [value for value in dummy if value != "Not even"]
    return dummy


def every_element_is_odd(input_list):
    count_ = 0
    for i in range(len(input_list)):
        if input_list[i] % 2 == 0:
            count_ += 1
        else:
            count_ += 0

    if count_ == 0:
        return True
    else:
        return False


def kth_largest_in_list(input_list, kth_largest):
    input_list.sort()
    return input_list[len(input_list)-kth_largest]


def cumavg_list(input_list):
    cumavg = [None] * len(input_list)
    for i in range(len(input_list)):
        cumavg[i] = sum(input_list[0:(i + 1)]) / (i + 1)

    return cumavg




def element_wise_multiplication(input_list1, input_list2):
    x = [None]*len(input_list1)
    for i in range(len(input_list1)):
        x[i]=input_list1[i]*input_list2[i]

    return x


def merge_lists(*lists):
    return [item for sublist in lists for item in sublist]


def squared_odds(input_list):
    oddlist = [value for value in input_list if value % 2 == 1]

    for i in range(len(oddlist)):
        oddlist[i] = oddlist[i] * oddlist[i]

    return oddlist




def reverse_sort_by_key(input_dict):
    return dict(sorted(input_dict.items(), reverse = True))


def sort_list_by_divisibility(input_list):
    superdict = dict()
    by_two_and_five = list()
    by_two = list()
    by_five = list()
    by_none = list()
    for i in range(len(input_list)):

        if input_list[i] % 10 == 0:
            by_two_and_five.append(input_list[i])
        else:
            if input_list[i] % 5 == 0:
                by_five.append(input_list[i])
            else:
                if input_list[i] % 2 == 0:
                    by_two.append(input_list[i])
                else:
                    by_none.append(input_list[i])

    superdict = {
        "by_two": by_two,
        "by_five": by_five,
        "by_two_and_five": by_two_and_five,
        "by_none": by_none
    }

    return superdict
