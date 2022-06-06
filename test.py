def gen_t():
    for i in range(10):
        yield i

def main():
    x = gen_t()
    for i in x:
        print(i)
    print("over")

if __name__ == "__main__":
    main()