"""Manual Jarvis cleanup helper.

This module stays import-safe so pytest can collect the repository without
requiring the optional ``jlclient`` dependency.
"""

import os


def main() -> None:
    from jlclient import jarvisclient
    from jlclient.jarvisclient import post

    token = os.environ.get("JARVIS_API_TOKEN")
    machine_id = os.environ.get("JARVIS_MACHINE_ID")
    if not token or not machine_id:
        raise RuntimeError("Set JARVIS_API_TOKEN and JARVIS_MACHINE_ID before running this helper.")

    jarvisclient.token = token
    destroy_response = post(
        {},
        "misc/destroy",
        jarvisclient.token,
        query_params={"machine_id": int(machine_id)},
    )
    # data = {"num_gpus": 1, "hdd": 40, "is_reserved": True, "name": "<name>", "disk_type": "ssd", "gpu_type": "<gpu_type>"}
    # res = post(data, "templates/vm/create", jarvisclient.token)
    # details = jarvisclient.Instance.destroy(<machine_id>)
    print(destroy_response)


if __name__ == "__main__":
    main()
