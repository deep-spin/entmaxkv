from setuptools import setup, find_packages

setup(
    name="entmaxkv",
    version="0.1.0",
    description="EntmaxKV: sparse entmax attention for KV cache compression",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5",
        "triton>=3.0",
    ],
    extras_require={
        "diagnostics": ["scipy"],
        "dev": ["pytest", "scipy"],
    },
)
