"""Access to the bundled agent skill file (SKILL.md)."""

import shutil
from pathlib import Path

__all__ = ["skill_path", "install_agent_skill"]


def skill_path():
    """Return the path to the SKILL.md bundled with the installed package.

    Returns
    -------
    pathlib.Path
        Path to the bundled SKILL.md, a complete usage skill for this
        package intended for AI agents.
    """
    return Path(__file__).parent / "SKILL.md"


def install_agent_skill(agent):
    """Install the bundled objscale agent skill for a supported agent framework.

    Copies the SKILL.md bundled with the installed package into the
    agent framework's skills directory:

    - ``'claude'``: ``~/.claude/skills/objscale/SKILL.md``
    - ``'codex'``: ``~/.codex/skills/objscale/SKILL.md``

    Directories are created as needed. For other agent frameworks, copy
    the file at ``skill_path()`` into the appropriate skills directory
    manually.

    Parameters
    ----------
    agent : str
        The agent framework to install the skill for. Must be
        ``'claude'`` or ``'codex'``.

    Returns
    -------
    pathlib.Path
        The destination path of the installed SKILL.md.

    Raises
    ------
    ValueError
        If ``agent`` is not ``'claude'`` or ``'codex'``.
    """
    skill_dirs = {
        "claude": Path.home() / ".claude" / "skills" / "objscale",
        "codex": Path.home() / ".codex" / "skills" / "objscale",
    }
    if agent not in skill_dirs:
        raise ValueError(
            "Unknown agent {!r}: must be 'claude' or 'codex'. For other "
            "agent frameworks, copy the file at objscale.skill_path() "
            "into the appropriate skills directory manually.".format(agent)
        )
    destination_dir = skill_dirs[agent]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "SKILL.md"
    shutil.copyfile(str(skill_path()), str(destination))
    print("Installed objscale agent skill to {}".format(destination))
    return destination
