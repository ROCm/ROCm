import re

from util.release_data import ReleaseLib
from util.defaults import TEMPLATES, PROCESSORS

TEMPLATES['MIVisionX'] = (
    (
        r"## MIVisionX (?P<lib_version>\d+\.\d+(?:\.\d+))"
        r"( \(Unreleased\))?"
        r"\n"
        r"(?P<body>(?:(?!## ).*(?:(?!\n## )\n|(?=\n## )))*)"
    )
)


def mivisionx_processor(data: ReleaseLib, template: str, _) -> bool:
    """Processor for MIVisionX releases."""
    changelog = data.repo.get_contents("CHANGELOG.md", data.commit)
    changelog = changelog.decoded_content.decode()
    pattern = re.compile(template)
    match = pattern.search(changelog)
    data.message = (
        f"MIVisionX for ROCm"
        f" {data.full_version}"
    )

    readme = data.repo.get_contents("README.md", data.commit)
    readme = readme.decoded_content.decode()
    dependency_map = readme[readme.find("## MIVisionX Dependency Map"):]
    data.notes = f"""
<p align="center">
    <img width="70%"
        src="https://github.com/ROCm/MIVisionX/raw/master/docs/images/MIVisionX.png" />
</p>

## Online Documentation
[MIVisionX Documentation](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html)
## MIVisionX {match['lib_version']}
{match["body"]}
{dependency_map}
"""
    return True


PROCESSORS['MIVisionX'] = mivisionx_processor
