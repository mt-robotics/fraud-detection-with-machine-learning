# pylint: disable=missing-module-docstring
import subprocess
import json
from collections import defaultdict


def main():
    """Generate license reports."""
    # Install pip-licenses if missing
    subprocess.run(["pip", "install", "pip-licenses"], check=True)

    # Generate full license report (DEPENDENCIES.md)
    with open("DEPENDENCIES.md", "w", encoding="utf-8") as f:
        subprocess.run(["pip-licenses", "--format=markdown"], stdout=f, check=True)

    # Generate simplified NOTICE.md
    with open("NOTICE.md", "w", encoding="utf-8") as f:
        f.write("# Third-Party Licenses\n\n")
        f.write("This project uses the following open-source components:\n\n")

        try:
            # Get all dependencies
            proc = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            deps = json.loads(proc.stdout)

            # Group by license type
            license_groups = defaultdict(list)
            for dep in deps:
                license_groups[dep["License"]].append(dep["Name"])

            # Write grouped licenses
            for license_type, packages in license_groups.items():
                f.write(f"### {license_type}\n")
                f.write(", ".join(sorted(packages)))  # Alphabetical order
                f.write("\n\n")

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Error generating license report: {e}")
            exit(1)


if __name__ == "__main__":
    main()
