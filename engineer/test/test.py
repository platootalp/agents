import types


def check(a, /, *args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == '__main__':
    L1 = ['Hello', 'World', 18, 'Apple', None]
    L2 = [x.lower() for x in L1 if isinstance(x, str)]
    L3 = [x.lower() if isinstance(x, str) else x for x in L1]

    # 测试:
    print(L2)
    print(L3)
    check("dood", 1, True, None, name="Alice", age=25, city="Beijing")
    # / 用于指示函数参数是位置参数（Positional - only），意味着它们只能通过位置传递，不能通过关键字传递。
