def if_true_test():
    for j in range(10):
        for i in range(5):
            print(i)
        if True == False:
            continue
        print("Still printing??")
        print(j)

if __name__ == '__main__':
    if_true_test()
