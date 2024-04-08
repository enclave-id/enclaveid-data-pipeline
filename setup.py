from setuptools import find_packages, setup

setup(
    name="enclaveid_data_pipeline",
    packages=find_packages(exclude=["enclaveid_data_pipeline_tests"]),
    install_requires=[
        "adlfs",
        "dagster>=1.6.0,<1.7.0",
        "dagster-cloud",
        "dagster-polars",
        "httpx",
        "mistralai",
        "numpy",
        "openai",
        "polars==0.20.15",
        "pandas",
        "pgvector",
        "psycopg[binary]",
        "pyarrow",
        "skypilot[azure]",
        "tqdm",
        "universal_pathlib",
    ],
    extras_require={
        "dev": [
            "dagster-webserver",
            "pytest",
            "ipython",
            "ipykernel",
            "ipywidgets",
            "ruff",
            "vllm",
            "sentence-transformers",
            "cupy-cuda12x",
            "hdbscan",
        ]
    },
)

# pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.2.*
