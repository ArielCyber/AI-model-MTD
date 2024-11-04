import random
import time

def randomize(arr,seed):
    if(seed == -1):
        t = int( time.time() * 1000.0 )
        random.seed( ((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24)   )
    else:
        random.seed(seed)
    return rand_array(arr)

def rand_array(arr):
    randomized_indices = random.sample(range(len(arr)), len(arr))
    randomized_array = [arr[i] for i in randomized_indices]
    inidices = []
    if type(arr[0]) == list:
        for i,index in enumerate(randomized_indices):
            new_arr, new_ind = rand_array(arr[index])
            randomized_array[i] = new_arr
            inidices.append((index,new_ind))
            randomized_array
    else:
        for i in randomized_indices:
            inidices.append((i,None))


    return randomized_array, inidices

def retrieve(randomized_array, randomized_indices):
    retrieved_array = [None] * len(randomized_array)
    for i, index in enumerate(randomized_indices):
        if index[1] is None:
            retrieved_array[index[0]] = randomized_array[i]
        else:
            retrieved_array[index[0]] = retrieve(randomized_array[i], index[1])
    return retrieved_array

"""new_values, map = randomize([[10,20,30,40,50,60],[11,21,31,41,51,61],[12,22,32,42,52,62],[13,23,33,43,53,63]],12425735)
print(new_values,map)
print(retrieve(new_values,map))"""