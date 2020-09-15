from . import position as pos


class Waypoint:
    """
    Class representing a waypoint for the drone to fly to
    """

    def __init__(self, pos):
        self.pos = pos

    def reached(self, drone_pos):
        return drone_pos == self.pos

    def reached_zy(self, drone_pos, error=0.1):
        return drone_pos.equal_zy(self.pos, error)

    def __str__(self):
        return str(self.pos)

if __name__ == "__name__":
    wp = Waypoint(pos.Position(0,0,0))
    assert wp.reached(pos.Position(0.1, 0, 0))
    assert not wp.reached(pos.Position(123, -2, 0))