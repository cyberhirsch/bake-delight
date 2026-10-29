"""bake-delight: command line for Bake & Delight.

    bake-delight new stone.bad --scan stone.obj --photo stone.jpg --set scan.faces=6000
    bake-delight set stone.bad scan.remesher=autoremesher scan.resolution=4096
    bake-delight info stone.bad
    bake-delight scan stone.bad              optimise the scan, store the result in the project
    bake-delight delight stone.bad           de-light the project's lit map
    bake-delight delight --lit lit.png --out albedo.png      no project: maps from the command line
    bake-delight keys                         every setting with type, default and choices
    bake-delight mcp                          MCP server on stdio, for Claude and other agents

Settings are section.key (bare keys work when unambiguous). --set on a run applies to that
run only; `set` stores it in the project. Exit status: 0 done, 1 failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bakedelight import pipeline
from bakedelight.project import SUFFIX, Project, ProjectError, describe_settings, split_key

RESULT_PREFIX = "result: "  # the last stdout line of a run with --json; the MCP server reads it


def parse_assignments(items: list[str]) -> dict:
    values = {}
    for item in items or []:
        if "=" not in item:
            raise ProjectError(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        split_key(key.strip())  # fail early on unknown keys
        values[key.strip()] = value
    return values


def open_project(path: str | None, args) -> Project:
    """The project to run: loaded from `path`, or built in memory from command-line inputs."""
    project = Project.load(path) if path else Project("command line")
    inputs = {key: getattr(args, key) for key in ("lit", "normal", "ao", "mask", "scan", "photo") if getattr(args, key, None)}
    values = {("delight." if key in ("lit", "normal", "ao", "mask") else "scan.") + key: value for key, value in inputs.items()}
    out = getattr(args, "out", None)
    if out and args.command == "delight":
        out = Path(out)
        values.update({"delight.output_dir": str(out.parent.resolve()), "delight.output_name": out.name})
    elif out:
        values["scan.out"] = out
    values.update(parse_assignments(args.set))
    project.update(values)
    if args.command == "delight":
        project.autofill_maps()
    return project


def emit(args, payload) -> None:
    if args.json:
        print(RESULT_PREFIX + json.dumps(payload), flush=True)
    else:
        print(json.dumps(payload, indent=2))


def command_new(args) -> int:
    path = Path(args.project)
    if path.suffix.lower() != SUFFIX:
        path = path.with_name(path.name + SUFFIX)
    if path.exists() and not args.force:
        raise ProjectError(f"{path} exists; pass --force to overwrite")
    project = Project(args.name or path.stem, path)
    project.update(parse_assignments(args.set))
    values = {f"delight.{key}": getattr(args, key) for key in ("lit", "normal", "ao", "mask") if getattr(args, key)}
    values.update({f"scan.{key}": getattr(args, key) for key in ("scan", "photo") if getattr(args, key)})
    project.update(values)
    project.autofill_maps()
    project.save()
    emit(args, project.summary())
    return 0


def command_info(args) -> int:
    project = Project.load(args.project)
    emit(args, project.to_dict(relative=False) if args.full else project.summary())
    return 0


def command_set(args) -> int:
    project = Project.load(args.project)
    project.update(parse_assignments(args.assignments))
    if args.notes is not None:
        project.notes = args.notes
    project.save()
    emit(args, project.summary())
    return 0


def command_keys(args) -> int:
    rows = describe_settings()
    if args.json:
        emit(args, rows)
        return 0
    for row in rows:
        choices = f"  {row['choices']}" if row["choices"] else ""
        print(f"{row['key']:<28} {row['type']:<6} {row['default']!r:<24} {row['description']}{choices}")
    return 0


def command_run(args) -> int:
    project = open_project(args.project, args)
    run = pipeline.delight if args.command == "delight" else pipeline.scan
    result = run(project, log=print)
    if args.project and not args.no_save:
        pipeline.merge_results(Path(args.project), args.command, result)
    emit(args, result)
    return 0


def command_mcp(_args) -> int:
    from bakedelight import mcp_server
    mcp_server.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bake-delight", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def common(sub):
        sub.add_argument("--json", action="store_true", help=f"print the outcome as one '{RESULT_PREFIX}<json>' line")
        return sub

    new = common(commands.add_parser("new", help="create a project file"))
    new.add_argument("project", help=f"project file ({SUFFIX} is added)")
    new.add_argument("--name")
    new.add_argument("--force", action="store_true", help="overwrite an existing project")

    info = common(commands.add_parser("info", help="show a project"))
    info.add_argument("project")
    info.add_argument("--full", action="store_true", help="every setting, with absolute paths")

    setting = common(commands.add_parser("set", help="change settings stored in a project"))
    setting.add_argument("project")
    setting.add_argument("assignments", nargs="*", metavar="key=value")
    setting.add_argument("--notes", help="replace the project notes")

    common(commands.add_parser("keys", help="list every setting"))

    for name, text in (("delight", "de-light a lit texture"), ("scan", "optimise a scan into a PBR asset")):
        run = common(commands.add_parser(name, help=text))
        run.add_argument("project", nargs="?", help="project file; omit to work from command-line inputs only")
        run.add_argument("--out", help="output file (delight) or folder (scan)")
        run.add_argument("--no-save", action="store_true", help="do not store the result in the project")
    for sub in (new, commands.choices["delight"]):
        for key in ("lit", "normal", "ao", "mask"):
            sub.add_argument(f"--{key}", help=f"{key} map" + ("; normal/ao/mask are found next to it" if key == "lit" else ""))
    for sub in (new, commands.choices["scan"]):
        sub.add_argument("--scan", help="high-res scan mesh")
        sub.add_argument("--photo", help="photo texture")
    for sub in (new, commands.choices["delight"], commands.choices["scan"]):
        sub.add_argument("--set", action="append", metavar="key=value", help="setting, repeatable")

    common(commands.add_parser("mcp", help="run the MCP server on stdio"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"new": command_new, "info": command_info, "set": command_set, "keys": command_keys,
               "delight": command_run, "scan": command_run, "mcp": command_mcp}[args.command]
    try:
        return handler(args)
    except ProjectError as error:
        print(f"bake-delight: {error}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as error:
        print(f"bake-delight: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
