file_path="classes.txt"

def read_classes():
    d={}
    i=0
    with open(file_path,'r') as filex:
        l=filex.readlines()
    l = [x.strip() for x in l]
    for x in l:
        if (i//10==0):
            index=int(x[-1])
            classx=x[:-2]
        else:
            index=int(x[-2:])
            classx=x[:-3]
        i+=1
        d[index]=classx
    # print(d)
    return d

# read_classes()