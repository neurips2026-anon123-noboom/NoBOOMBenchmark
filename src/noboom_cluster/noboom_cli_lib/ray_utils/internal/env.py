from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import string
from typing import Dict, Mapping, Tuple

from dotenv import dotenv_values, set_key


def _rand_alnum(n: int) -> str:
    """Generate a random alphanumeric string.

    Args:
        n (int): Length of the string to generate.

    Returns:
        str: Random alphanumeric string of length ``n``.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def write_seaweedfs_s3_auth_json(
    path: str | os.PathLike,
    *,
    identity_name: str = "ray",
    access_key_len: int = 24,
    secret_key_len: int = 48,
    actions: Tuple[str, ...] = ("Read", "Write", "List", "Tagging", "Admin"),
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Create and write a SeaweedFS S3 auth JSON file.

    Args:
        path (str | os.PathLike): Output JSON filepath, e.g. "./s3.json".
        identity_name (str): Identity name stored in JSON. Defaults to "ray".
        access_key_len (int): Length of the access key. Defaults to 24.
        secret_key_len (int): Length of the secret key. Defaults to 48.
        actions (Tuple[str, ...]): Allowed actions for this identity.
        overwrite (bool): Whether to overwrite if the file exists. Defaults to False.

    Returns:
        Tuple[str, str]: Generated (access_key, secret_key) credentials.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.

    Side Effects:
        Writes a JSON file to disk and performs an atomic replace.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists (set overwrite=True to replace).")

    out.parent.mkdir(parents=True, exist_ok=True)

    access_key = _rand_alnum(access_key_len)
    secret_key = secrets.token_urlsafe(max(1, secret_key_len // 2))  # ~1.33x bytes -> chars
    # Ensure minimum length (token_urlsafe length is approximate)
    if len(secret_key) < secret_key_len:
        secret_key += secrets.token_urlsafe((secret_key_len - len(secret_key) + 1) // 2)
        secret_key = secret_key[:secret_key_len]

    data: Dict[str, object] = {
        "identities": [
            {
                "name": identity_name,
                "credentials": [{"accessKey": access_key, "secretKey": secret_key}],
                "actions": list(actions),
            }
        ]
    }

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, out)

    return access_key, secret_key


_VAR_REF_RE = re.compile(r"\$(\w+)|\$\{(\w+)}")


def update_env(
    env_path: str | Path,
    new_vars: dict[str, str],
    suffix: str = "",
    *,
    max_passes: int = 20,
    strict_undefined: bool = False,
) -> str:
    """Update a .env file without directly modifying ``os.environ``.

    Args:
        env_path (str | Path): Path to the .env file to update.
        new_vars (dict[str, str]): New variables to inject or override.
        suffix (str): Suffix to apply when writing the new .env file. Defaults to "".
        max_passes (int): Max passes for resolving forward references. Defaults to 20.
        strict_undefined (bool): Whether to raise on unresolved variables. Defaults to False.

    Returns:
        str: Path to the updated .env file.

    Raises:
        ValueError: If unresolved variables remain when ``strict_undefined`` is True.

    Side Effects:
        Writes an updated .env file to disk.
    """
    env_path = Path(env_path)

    if suffix:
        # Create a suffixed copy (your existing behavior)
        new_env_path = env_path.with_suffix("")
        new_env_path = new_env_path.with_name(f"{new_env_path.name}__{suffix}")
        new_env_path.write_text(env_path.read_text())
        env_path = str(new_env_path)
    else:
        env_path = str(env_path)

    # Read existing variables (and allow system env as fallback, as your code does)
    file_vars = os.environ | dotenv_values(env_path)  # type: ignore[assignment]
    lookup: dict[str, str] = {k: v for k, v in file_vars.items() if v is not None}

    def expand_with(local_lookup: Mapping[str, str], value: str) -> str:
        """Expand $VAR or ${VAR} using a local lookup mapping.

        Args:
            local_lookup (Mapping[str, str]): Variable lookup table.
            value (str): Template string to expand.

        Returns:
            str: Expanded string with substitutions applied.
        """

        def repl(m: re.Match[str]) -> str:
            """Resolve a single regex match to a variable value.

            Args:
                m (re.Match[str]): Regex match containing variable name groups.

            Returns:
                str: Resolved value from the lookup or an empty string.
            """
            name = m.group(1) or m.group(2)
            return local_lookup.get(name, "")

        return _VAR_REF_RE.sub(repl, value)

    # ---- Fixed-point resolution for new_vars (order independent) ----
    templates: dict[str, str] = {k: str(v) for k, v in new_vars.items()}
    resolved: dict[str, str] = {}

    # Start with existing lookup, then refine resolved values across passes.
    # Each pass uses: base lookup + latest resolved values.
    for _ in range(max_passes):
        changed = False
        current_lookup = {**lookup, **resolved}

        for k, tmpl in templates.items():
            val = expand_with(current_lookup, tmpl)
            if resolved.get(k) != val:
                resolved[k] = val
                changed = True

        if not changed:
            break

    if strict_undefined:
        # If strict, check if any variable reference remains that we cannot satisfy.
        # This catches cycles or missing vars.
        final_lookup = {**lookup, **resolved}
        unresolved: dict[str, set[str]] = {}
        for k, tmpl in templates.items():
            refs = {a or b for (a, b) in _VAR_REF_RE.findall(tmpl)}
            missing = {r for r in refs if r and r not in final_lookup}
            if missing:
                unresolved[k] = missing
        if unresolved:
            raise ValueError(f"Unresolved .env references after {max_passes} passes: {unresolved}")

    # Persist in a stable order (optional but nice for diffs)
    for key in sorted(resolved.keys()):
        set_key(env_path, key, resolved[key], quote_mode="never")
        lookup[key] = resolved[key]  # keep lookup consistent

    return env_path
