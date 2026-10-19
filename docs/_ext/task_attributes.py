"""Sphinx extension that documents the attributes of an ESTEEM task class.

Usage in a .rst file::

    .. task-attributes:: esteem.tasks.solutes.SolutesTask

       wrapper: The wrapper used to run the calculations, eg ``ORCAWrapper()``.
       script_settings: Settings for the job scripts written for this task.

The table is built from the task's argument parser (``make_parser``), whose
arguments set the attributes of the task object. Each line of the directive's
content adds a row for an attribute that is not in the parser, in the form
``name: description`` (the description may contain reStructuredText).
"""

import argparse
import importlib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


class TaskAttributes(Directive):
    required_arguments = 1
    has_content = True

    def _cell(self, content):
        entry = nodes.entry()
        if isinstance(content, nodes.Node):
            entry += content
        else:
            entry += nodes.paragraph(text=content)
        return entry

    def _row(self, *cells):
        row = nodes.row()
        for cell in cells:
            row += self._cell(cell)
        return row

    def _parsed(self, text):
        # Parse a description written in reStructuredText
        container = nodes.container()
        self.state.nested_parse(StringList([text]), self.content_offset, container)
        return container.children[0] if len(container.children) == 1 else container

    def run(self):
        modname, clsname = self.arguments[0].rsplit('.', 1)
        cls = getattr(importlib.import_module(modname), clsname)
        task = cls()
        actions = [a for a in task.make_parser()._actions
                   if not isinstance(a, argparse._HelpAction)]

        # Use a typical setting (not 'seed', which the drivers set) as the example
        example = next((a.dest for a in actions if a.dest != 'seed'), 'seed')
        intro = self._parsed(
            f'These settings are attributes of a ``{clsname}`` object: set them in a workflow '
            f'script, for example ``task.{example} = ...``, before passing the task to '
            f'``drivers.main``. When the module is run from the command line, each one in the '
            f'first table can also be given as an option of the same name, for example '
            f'``--{example}``.')

        table = nodes.table(classes=['colwidths-given', 'task-attributes'])
        tgroup = nodes.tgroup(cols=3)
        table += tgroup
        for width in (30, 12, 58):
            tgroup += nodes.colspec(colwidth=width)
        thead = nodes.thead()
        thead += self._row('Attribute', 'Default', 'Description')
        tgroup += thead
        tbody = nodes.tbody()
        tgroup += tbody

        for action in actions:
            helptext = action.help if action.help not in (None, argparse.SUPPRESS) else ''
            # The argparse type is left out: it describes command-line parsing, and scripts
            # can often set other types (dicts, lists, objects), as the descriptions explain
            tbody += self._row(nodes.literal(text=action.dest),
                               nodes.literal(text=repr(action.default)), helptext)

        result = [intro, table]

        # Extra attributes that are not command-line arguments
        extra_rows = []
        for line in self.content:
            if not line.strip():
                continue
            name, _, desc = line.partition(':')
            name = name.strip()
            default = repr(getattr(task, name)) if hasattr(task, name) else ''
            extra_rows.append(self._row(nodes.literal(text=name),
                                        nodes.literal(text=default) if default else '',
                                        self._parsed(desc.strip())))
        if extra_rows:
            result.append(self._parsed(
                'The following attributes are set only from a workflow script, '
                'not from the command line:'))
            table = nodes.table(classes=['colwidths-given', 'task-attributes'])
            tgroup = nodes.tgroup(cols=3)
            table += tgroup
            for width in (30, 12, 58):
                tgroup += nodes.colspec(colwidth=width)
            thead = nodes.thead()
            thead += self._row('Attribute', 'Default', 'Description')
            tgroup += thead
            tbody = nodes.tbody()
            for row in extra_rows:
                tbody += row
            tgroup += tbody
            result.append(table)

        return result


def setup(app):
    app.add_directive('task-attributes', TaskAttributes)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
