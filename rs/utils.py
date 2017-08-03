import os
import pprint
import configparser
import pandas as pd
from numbers import Number
from functools import partial
from inspect import isroutine
from tabulate import tabulate

def pretty_print(data, description='■ Recommendations:'):
    """Print pandas data frame as table or default Python to pretty printer
    """
    print(description)
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data)
        print(tabulate(df, headers='keys', tablefmt='orgtbl'))
    elif isinstance(data, pd.DataFrame):
        print(tabulate(data, headers='keys', tablefmt='orgtbl'))
    else:
        pp = pprint.PrettyPrinter()
        pp.pprint(data)


def parse_config(section, key, ini_file_path='../config.ini'):
    """Parse the configuration file and return option value associated with a key in a specified section
    """
    def in_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    config = configparser.ConfigParser()
    ini_file = 'config.ini' if in_ipython() else '../config.ini'
    config.read(ini_file)
    contents = config_section_map(config, section)
    if key not in contents:
        raise ValueError('Invalid key provided')
    return contents[key]


def config_section_map(config, section):
    """Get the contents of a configuration section as a dictionary
    """
    data = {}
    options = config.options(section)
    for option in options:
        try:
            data[option] = config.get(section, option)
            if data[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            data[option] = None
    return data


def get_dump_path(name):
    """Get the path where the dumped model binary file will be stored
    """
    base = parse_config('Path', 'trained')
    path = os.path.expanduser(base + '/' + name)
    return path


def print_object(obj):
    """Print object properties in formatted style
    """
    print(ppretty_obj(obj,
          seq_length=10,
          show_protected=True,
          show_static=True,
          show_properties=True,
          show_address=True))


def ppretty_obj(obj, indent='    ', depth=4, width=72, seq_length=5,
                show_protected=False, show_private=False, show_static=False,
                show_properties=False, show_address=False, str_length=50):
    """Source: https://github.com/symonsoft/ppretty
    """
    seq_brackets = {list: ('[', ']'), tuple: ('(', ')'), set: ('set([', '])'), dict: ('{', '}')}
    seq_types = tuple(seq_brackets.keys())

    def inspect_object(current_obj, current_depth, current_width, seq_type_descendant=False):
        inspect_nested_object = partial(inspect_object,
                                        current_depth=current_depth - 1,
                                        current_width=current_width - len(indent))

        if isinstance(current_obj, Number):
            return [repr(current_obj)]

        if isinstance(current_obj, str):
            if len(current_obj) <= str_length:
                return [repr(current_obj)]
            return [repr(current_obj[:int(str_length / 2)] + '...' + current_obj[int((1 - str_length) / 2):])]

        if isinstance(current_obj, type):
            md = current_obj.__module__ + '.' if hasattr(current_obj, '__module__') else ''
            return ["<class '" + md + current_obj.__name__ + "'>"]

        if current_obj is None:
            return ['None']

        def format_block(lines, open_bkt='', close_bkt=''):
            one_line = ''
            new_lines = []
            if open_bkt:
                new_lines.append(open_bkt)
                one_line += open_bkt
            for line in lines:
                new_lines.append(indent + line)
                if len(one_line) <= current_width:
                    one_line += line
            if close_bkt:
                if lines:
                    new_lines.append(close_bkt)
                else:
                    new_lines[-1] += close_bkt
                one_line += close_bkt

            return [one_line] if len(one_line) <= current_width and one_line else new_lines

        class SkipElement(object):
            pass

        class ErrorAttr(object):
            def __init__(self, e):
                self.e = e

        def cut_seq(seq):
            if current_depth < 1:
                return [SkipElement()]
            if len(seq) <= seq_length:
                return seq
            elif seq_length > 1:
                seq = list(seq) if isinstance(seq, tuple) else seq
                return seq[:int(seq_length / 2)] + [SkipElement()] + seq[int((1 - seq_length) / 2):]
            return [SkipElement()]

        def format_seq(extra_lines):
            r = []
            items = cut_seq(obj_items)
            for n, i in enumerate(items, 1):
                if type(i) is SkipElement:
                    r.append('...')
                else:
                    if type(current_obj) is dict or seq_type_descendant and isinstance(current_obj, dict):
                        (k, v) = i
                        k = inspect_nested_object(k)
                        v = inspect_nested_object(v)
                        k[-1] += ': ' + v.pop(0)
                        r.extend(k)
                        r.extend(format_block(v))
                    elif type(current_obj) in seq_types or seq_type_descendant and isinstance(current_obj, seq_types):
                        r.extend(inspect_nested_object(i))
                    else:
                        (k, v) = i
                        k = [k]
                        v = inspect_nested_object(v) if type(v) is not ErrorAttr else ['<Error attribute: ' + type(v.e).__name__ + ': ' + v.e.message + '>']
                        k[-1] += ' = ' + v.pop(0)
                        r.extend(k)
                        r.extend(format_block(v))
                if n < len(items) or extra_lines:
                    r[-1] += ', '
            return format_block(r + extra_lines, *brackets)

        extra_lines = []
        if type(current_obj) in seq_types or seq_type_descendant and isinstance(current_obj, seq_types):
            if isinstance(current_obj, dict):
                obj_items = current_obj.items()
            else:
                obj_items = current_obj

            if seq_type_descendant:
                brackets = seq_brackets[[seq_type for seq_type in seq_types if isinstance(current_obj, seq_type)].pop()]
            else:
                brackets = seq_brackets[type(current_obj)]
        else:
            obj_items = []
            for k in sorted(dir(current_obj)):
                if not show_private and k.startswith('_') and '__' in k:
                    continue
                if not show_protected and k.startswith('_'):
                    continue
                try:
                    v = getattr(current_obj, k)
                    if isroutine(v):
                        continue
                    if not show_static and hasattr(type(current_obj), k) and v is getattr(type(current_obj), k):
                        continue
                    if not show_properties and hasattr(type(current_obj), k) and isinstance(
                            getattr(type(current_obj), k), property):
                        continue
                except Exception as e:
                    v = ErrorAttr(e)

                obj_items.append((k, v))

            if isinstance(current_obj, seq_types):
                extra_lines += inspect_nested_object(current_obj, seq_type_descendant=True)

            md = current_obj.__module__ + '.' if hasattr(current_obj, '__module__') else ''
            address = ' at ' + hex(id(current_obj)) + ' ' if show_address else ''
            brackets = (md + type(current_obj).__name__ + address + '(', ')')

        return format_seq(extra_lines)

    return '\n'.join(inspect_object(obj, depth, width))
