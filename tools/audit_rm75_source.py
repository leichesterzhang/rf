"""Export RM75 configuration inheritance and source evidence without Isaac Gym.

Run from any directory with Python 3.8+. Only trusted local configuration modules
are imported; environment, training, torch and simulator modules are not executed.
This is a source audit, not a simulation or checkpoint validation.
"""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "source_audit"


def plain(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    return {k: plain(getattr(obj, k)) for k in dir(obj)
            if not k.startswith("_") and not inspect.isroutine(getattr(obj, k))}


def source_link(path, line):
    return "[{}:{}]({}:{})".format(path.relative_to(ROOT).as_posix(), line,
                                    path.as_posix(), line)


def main():
    sys.dont_write_bytecode = True
    # Package shells prevent importing envs/__init__.py (which loads Isaac Gym).
    for name in ("legged_gym", "legged_gym.envs", "legged_gym.envs.base",
                 "legged_gym.envs.go2", "legged_gym.envs.rm75"):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules[name] = module

    registry_path = ROOT / "legged_gym/envs/__init__.py"
    registry = ast.parse(registry_path.read_text(encoding="utf-8-sig"))
    symbols = {}
    for node in registry.body:
        if isinstance(node, ast.ImportFrom) and node.module and ".rm75." in node.module and "config" in node.module:
            module = importlib.import_module(node.module)
            for alias in node.names:
                symbols[alias.asname or alias.name] = getattr(module, alias.name)

    tasks = {}
    for node in ast.walk(registry):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "register":
            continue
        name = ast.literal_eval(node.args[0])
        if not name.startswith("RM75"):
            continue
        env_cls = symbols[node.args[2].func.id]
        train_cls = symbols[node.args[3].func.id]
        tasks[name] = {"environment_class": node.args[1].id,
                       "registry_line": node.lineno,
                       "env_config_class": env_cls.__name__,
                       "train_config_class": train_cls.__name__,
                       "env": plain(env_cls()), "train": plain(train_cls())}

    classes = {}
    paths = list((ROOT / "legged_gym/envs").rglob("*.py"))
    paths += list((ROOT / "rsl_rl/rsl_rl").rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (path, node)

    def method(cls_name, method_name):
        if cls_name not in classes:
            return None
        path, node = classes[cls_name]
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                return path, child
        for base in node.bases:
            if isinstance(base, ast.Name):
                found = method(base.id, method_name)
                if found:
                    return found
        return None

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "effective_configs.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# RM75 有效配置与奖励源码证据", "",
          "由 tools/audit_rm75_source.py 生成；解析配置继承，未启动仿真。权重为配置原值，运行时通常还乘策略步长和奖励课程系数。", ""]
    used = {}
    for name, task in tasks.items():
        env, train = task["env"], task["train"]
        md += ["## " + name, "", "环境：`{}`；Actor：`{}`；算法：`{}`。".format(task["environment_class"], train["runner"]["policy_class_name"], train["runner"]["algorithm_class_name"]), "",
               "| 奖励 | 配置权重 | 实际实现 |", "|---|---:|---|"]
        for name_r, weight in env["rewards"]["scales"].items():
            if not weight:
                continue
            found = method(task["environment_class"], "_reward_" + name_r)
            if not found:
                raise RuntimeError("Missing reward: {} / {}".format(name, name_r))
            path, node = found
            used[(str(path), node.lineno)] = (path, node)
            md.append("| {} | {} | {} |".format(name_r, weight, source_link(path, node.lineno)))
        md += ["", "配置类位置：" + source_link(Path(inspect.getsourcefile(symbols[task["env_config_class"]])), inspect.getsourcelines(symbols[task["env_config_class"]])[1]), ""]
    md += ["# 奖励函数原文（去重）", ""]
    for path, node in used.values():
        src = path.read_text(encoding="utf-8-sig").splitlines()
        md += ["## " + node.name, "", source_link(path, node.lineno), "", "```python",
               "\n".join(src[node.lineno - 1:node.end_lineno]), "```", ""]
    (OUT / "reward_evidence.md").write_text("\n".join(md), encoding="utf-8")

    assets = []
    for folder in ("RM75", "Z1_NOLIDARV1.1.SLDASM", "resources/robots/RM75"):
        for path in ROOT.joinpath(folder).rglob("*.urdf"):
            root = ET.parse(path).getroot()
            joints = [{"name": j.get("name"), "type": j.get("type")} for j in root.findall("joint")]
            assets.append({"path": path.relative_to(ROOT).as_posix(),
                           "joint_types": dict(Counter(j["type"] for j in joints)),
                           "joints": joints,
                           "summed_link_mass_kg": sum(float(m.get("value")) for m in root.findall("link/inertial/mass"))})
    (OUT / "asset_inventory.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")
    evidence_paths = set(paths + [registry_path])
    evidence_paths.update(ROOT / a["path"] for a in assets)
    for directory in ("legged_gym/utils", "legged_gym/scripts", "deploy/deploy_mujoco", "tools"):
        evidence_paths.update(ROOT.joinpath(directory).rglob("*.py"))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence_paths)}
    (OUT / "source_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("Audited {} RM75 tasks; {} unique active reward implementations; {} URDFs.".format(len(tasks), len(used), len(assets)))
    for name, task in tasks.items():
        print(name, "obs=", task["env"]["env"]["num_observations"], "critic=", task["env"]["env"]["num_privileged_obs"], "gravity=", task["env"]["sim"]["gravity"])


if __name__ == "__main__":
    main()
