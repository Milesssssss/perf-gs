# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Map the source shader tree into the installed trainer package."""
import sys
from tempfile import TemporaryDirectory

from setuptools import find_packages, setup

# Setuptools copies the entire build/lib tree into a wheel, including files
# left over from previous versions. Editable installs keep their normal build paths.
with TemporaryDirectory(prefix="perfgs-wheel-") as wheel_stage:
    options = {}
    if "bdist_wheel" in sys.argv:
        if "--skip-build" in sys.argv:
            raise SystemExit("PERF-GS wheels require a fresh build. Remove --skip-build.")
        options["build"] = {"build_base": wheel_stage}
    setup(
        packages=find_packages("src") + ["trainer.shaders"],
        package_dir={"": "src", "trainer.shaders": "shaders"},
        include_package_data=False,
        package_data={"trainer.shaders": ["**/*.slang"]},
        options=options,
    )
