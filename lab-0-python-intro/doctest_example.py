
def sum(*xs):
    """
    Sum the value of all arguments
    
    >>> sum(1)
    1
    >>> sum(1, 2)
    3
    >>> sum(3, 0, 8)
    11
    """
    sum = 0
    for x in xs:
        sum += x
    return sum

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)