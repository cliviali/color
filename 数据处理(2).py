import numpy as np

def comb( r,data):
    n = data.shape[0]
    if r > n:
        return 0
    else:
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if i == 0 and j == 1:
                    list1 = data[i]
                    list2 = data[j]
                    color_vector = np.concatenate([list1[3: ], list2[3: ]])
                    color_recipe_vector = list2[0: 3]-list1[0: 3]
                    array1 = np.expand_dims(np.concatenate([color_vector, color_recipe_vector]), axis=0)
                    result  = array1
                else:
                    list1 = data[i]
                    list2 = data[j]
                    color_vector = np.concatenate([list1[3: ], list2[3: ]])
                    color_recipe_vector = list2[0: 3]-list1[0: 3]
                    array1 = np.expand_dims(np.concatenate([color_vector, color_recipe_vector]), axis=0)
                    result =np.concatenate((result, array1))
        return  result



def comb2( r,data,lab):
    n = data.shape[0]
    if r > n:
        return 0
    else:
        result = []
        for i in range(n):
                if i == 0 :
                    list1 = data[i]
                    color_vector = np.expand_dims( np.concatenate([list1[3: ],lab]), axis=0)
                    result  = color_vector
                else:
                    list1 = data[i]
                    color_vector = np.expand_dims( np.concatenate([list1[3: ],lab]), axis=0)
                    result =np.concatenate((result, color_vector), axis=0)
        return  result
