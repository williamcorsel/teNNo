'''
Util functions
'''

from math import sqrt

def distance(p1, p2):
    '''
    Distance between the point p1 and p2
    Points are in the form (x,y)
    '''
    return sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))