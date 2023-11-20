class myfunctions:
    def __init__(self):
        pass
    def my_all(self, iterable):
        for element in iterable:
            if not element: #
                return False
        return True
    def my_abs(self, number):
        if number < 0:
            return -1 * (number // 1)
        return number
    def my_round(self, number, ndigits = None):
        if ndigits:
            if (number // 10 ** -ndigits) * (10 ** -ndigits) - (number // 10 ** -ndigits - 1) * (10 ** -ndigits - 1) > 5 * (10 ** -ndigits):
                return (number // 10 ** -ndigits) * (10 ** -ndigits) + 10 ** (-ndigits)
            return (number // 10 ** -ndigits) * (10 ** -ndigits)
        if ndigits == None:
            if (number - 1) > 0.5:
                return 2
            return 1
    def my_round_answer(self, number, ndigits = None):
        if ndigits is None:
            return int(number) + int((number - int(number)) * 2) #1 + another 1 or not, rounding algorithm
        return self.my_round_answer(number * 10 ** ndigits) / 10 ** ndigits #after multiplying 10^ndigits, do rounding
    def my_any(self, iterable):
        for element in iterable:
            if element: 
                return True
        return False
    def my_enumerate(self, sequence, start = 0):
        i = start
        for element in sequence:
            yield i , element 
            i += 1
    def my_max(self, *args):
        if len(args) == 1: #single iterable as an input
            return self.my_max(*list(args[0])) 
        max_value = args[0]    
        for value in args[1:]:
            if value > max_value:
                max_value = value
        return max_value
    def my_min(self, *args):
        if len(args) == 1:
            return self.my_min(*list(args[0]))
        min_value = args[0]    
        for value in args[1:]:
            if value < min_value:
                min_value = value
        return min_value
    def my_range(self, *args):
        #setting parameters
        if len(args) == 1:
            start, end, step = 0, args[0], 1 
        elif len(args) == 2:
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            start, end, step = args
        #when start is bigger than end, return nothing    
        if (end - start) * step < 0: 
            return []
        prev = end - start 
        while True:
            yield start
            start += step
            if (end - start) * prev <= 0: #end - start approaches zero
                break
    def my_reversed(self, seq):
        return list(seq)[::-1]
    def my_filter(self, function, iterable):
        for elem in iterable:
            if function(elem): #if input results True
                yield elem
    def my_map(self, function, iterable):
        for elem in iterable:
            yield function(elem)
    def my_sum(self, iterable, start=0):
        aggregated = type(start)(start) #copying input with explicit conversion 
        for elem in iterable:
            aggregated += elem
        return aggregated
    def my_zip(self, *iterables):
        iterators = [*myfunctions.my_map(iter, iterables)] #makes iterators from iterables
        last = object() 
        while True:
            #next() uses iterators, returning each item. when all elements are given, then iterators are exhausted 
            #also last as second parameter will be iterated after all elements are given 
            output = tuple(next(iterated, last) for iterated in iterators) #makes tuples of elements for each iterator 
            if last in output: #if all elements are give then break
                break
            yield output
if __name__ == "__main__":
    myfunctions = myfunctions() #class declaration

    test1 = [True, 7 == 4, 3 > 7, False]
    test2 = [3 < 5, 10 == 10, True, 'something']


    assert all(test1) == myfunctions.my_all(test1) == False
    assert all(test2) == myfunctions.my_all(test2) == True

    print("통과")

    test1 = 1.7
    test2 = -8

    assert abs(test1) == myfunctions.my_abs(test1)
    assert abs(test2) == myfunctions.my_abs(test2)

    print("통과")

    test = 1.74789

    assert round(test) == myfunctions.my_round_answer(test)
    assert round(test, 3) == myfunctions.my_round_answer(test, 3)
    assert round(-test, 2) == myfunctions.my_round_answer(-test, 2)

    print("통과")

    test1 = [True, 7 == 4, 'Something', False]
    test2 = [3 > 5, 10 != 10, False, '', None]
    assert any(test1) == myfunctions.my_any(test1)
    assert any(test2) == myfunctions.my_any(test2)
    
    print("통과")

    test1 = [60, 50, 20, 10]
    test2 = [True, None, 'test']

    assert list(enumerate(test1)) == list(myfunctions.my_enumerate(test1))
    assert list(enumerate(test2, 12)) == list(myfunctions.my_enumerate(test2, 12))

    print("통과")
    
    test = [7, 4, 2, 6, 8]
    
    assert max(test) == myfunctions.my_max(test) and min(test) == myfunctions.my_min(test)
    assert max(7, 4, 2, 5) == myfunctions.my_max(7, 4, 2, 5) and min(7, 4, 2, 5) == myfunctions.my_min(7, 4, 2, 5)
    
    print("통과")

    assert list(range(10)) == list(myfunctions.my_range(10))
    assert list(range(3, 10)) == list(myfunctions.my_range(3, 10))
    assert list(range(3, 10, 2)) == list(myfunctions.my_range(3, 10, 2))
    assert list(range(3, -10, -2)) == list(myfunctions.my_range(3, -10, -2))
    
    print("통과")

    test = [7, 4, 2, 6, 8]
    
    assert list(reversed(test)) == list(myfunctions.my_reversed(test))
    
    print("통과")

    def test_function(number):
        return number > 5
    
    test = [1, 7, 5, 2, 9, 11]


    assert list(filter(test_function, test)) == list(myfunctions.my_filter(test_function, test))\
    == list(filter(lambda x: x > 5, test))
    
    print("통과")

    def test_function(number):
        return number * 2
    
    test = [1, 7, 5, 2, 9, 11]

    assert list(map(test_function, test)) == list(myfunctions.my_map(test_function, test))\
    == list(map(lambda x: x * 2, test))
    
    print("통과")

    test = [7, 4, 2, 6, 8]

    assert sum(test) == myfunctions.my_sum(test)
    assert sum(range(10)) == myfunctions.my_sum(myfunctions.my_range(10))
    assert sum(filter(lambda x: x % 2, range(1, 20, 3)))\
    == myfunctions.my_sum(myfunctions.my_filter(lambda x: x % 2, myfunctions.my_range(1, 20, 3)))
    assert sum([[1, 2, 3], [5, 6, 7], [8, 9, 10]], start=[])\
    == myfunctions.my_sum([[1, 2, 3], [5, 6, 7], [8, 9, 10]], start=[]) 
    
    print("통과")


    test1 = (1, 2, 3)
    test2 = (10, 2, 30)
    
    assert list(zip(test1, test2)) == list(myfunctions.my_zip(test1, test2))

    test3 = [(10, 20, 30, 40), (55, 22, 66, 70), (False, True, True, False)]
    
    assert list(zip(*test3)) == list(myfunctions.my_zip(*test3))
    
    print("통과")