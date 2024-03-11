from setuptools import find_packages, setup

setup(
    name="enclaveid_data_pipeline",
    packages=find_packages(exclude=["enclaveid_data_pipeline_tests"]),
    install_requires=[
        "adlfs",
        "dagster>=1.6.0,<1.7.0",
        "dagster-cloud",
        "dagster-polars",
        "polars",
        "pandas",
        "pyarrow",
        "numpy",
        "openai",
        "httpx",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "dagster-webserver",
            "pytest",
            "ipython",
            "ipykernel",
        ]
    },
)
