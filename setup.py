# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Map the source shader tree into the installed trainer package."""
from setuptools import find_packages, setup

setup(
    packages=find_packages("src") + ["trainer.shaders"],
    package_dir={"": "src", "trainer.shaders": "shaders"},
    package_data={"trainer.shaders": ["**/*.slang"]},
)
