"""
    异步编程
"""
import asyncio


async def hello_world():
    await asyncio.sleep(1)
    print("Hello, world!")



def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()



if __name__ == '__main__':
    # coro = hello_world()
    # print(hello_world)  # <function hello_world at 0x102a93e20>
    # print(coro.__class__)  # <class 'coroutine'>
    # asyncio.run(coro)  # Hello, world!’
    c = consumer()
    produce(c)
    print(dir(c))
