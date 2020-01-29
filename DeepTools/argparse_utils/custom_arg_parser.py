import argparse

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwrags):
        super(CustomArgumentParser, self).__init__(*args, **kwrags)

    def convert_arg_line_to_args(self, arg_line):
        for arg in arg_line.split():
            if arg[0] == "#":
                break
            if not arg.strip():
                continue
            yield arg