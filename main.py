import sys

from transform_state import NonblindDeblurringTransformState

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python3 main.py <LEVEL_NO>")
        quit()
    
    level = int(sys.argv[1])

    if level == 1:
        vis = NonblindDeblurringTransformState()
        vis()