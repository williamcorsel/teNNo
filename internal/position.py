
error = 0.1

class Position:
    """
    Class for representing a position in 3d space
    Contains the x,y and z coordinates
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


    def equal_zy(self, other, error):
        if self.x > other.x and abs(self.y - other.y) <= error and abs(self.z - other.z) <= error:
            return True
        return False


    def __str__(self):
        return str(round(self.x, 2)) + "," + str(round(self.y, 2)) + "," + str(round(self.z, 2))


    def __eq__(self, other):
        if abs(self.x - other.x) <= error and abs(self.y - other.y) <= error and abs(self.z - other.z) <= error:
            return True
        return False


if __name__ == '__main__':
    pos1 = Position(0,0,0)
    pos2 = Position(0.1, 0, 0)
    pos3 = Position(0,0.5, 0)

    assert pos1 == pos2
    assert not pos1 == pos3