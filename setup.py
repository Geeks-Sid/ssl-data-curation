# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


setup(
    name="ssl_data_curation",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "patchselect=patchselect.__main__:main",
            "patchselect-benchmark=patchselect.benchmark_backends:main",
            "patchselect-export=patchselect.export_images:main",
            "patchselect-local=patchselect.run_local_selection:main",
            "patchselect-global=patchselect.run_global_selection:main",
            "patchselect-pack-tars=patchselect.export_tars:main",
        ]
    },
)
