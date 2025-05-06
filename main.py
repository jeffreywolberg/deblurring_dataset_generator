import sys

from level_one_transform_vis import LevelOneTransformVis
from level_three_transform_vis import LevelThreeTransformVis
from level_two_transform_vis import LevelTwoTransformVis


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python3 main.py <LEVEL_NO>")
        quit()
    
    level = int(sys.argv[1])

    if level == 1:
        vis = LevelOneTransformVis()
    elif level == 2:
        vis = LevelTwoTransformVis()
    elif level == 3:
        vis = LevelThreeTransformVis()
    else:
        raise NotImplementedError(f"Level {level} is not implemented yet")
    
    vis()