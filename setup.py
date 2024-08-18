from setuptools import setup, find_packages  # type: ignore

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="terpeneminer",
    version="0.1.0",
    author="Raman Samusevich",
    author_email="raman.samusevich@gmail.com",
    description="A package for highly accurate detection of terpene synthases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamusRam/TerpeneMiner",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "get_uniprot_sample=terpeneminer.src.data_preparation.get_uniprot_sample:main",
            "get_phylogeny_based_clusters=terpeneminer.src.data_preparation.get_phylogeny_based_clusters:main",
            "gather_plm_embeddings=terpeneminer.src.embeddings_extraction.gather_required_embs:main",
            "plm_embeddings=terpeneminer.src.embeddings_extraction.transformer_embs:main",
            "terpene_miner_main=terpeneminer.src.terpene_miner_main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.10.0",
)
