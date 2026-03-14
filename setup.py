from setuptools import setup, find_packages  # type: ignore

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enzymeexplorer",
    version="0.1.0",
    author="Raman Samusevich",
    author_email="raman.samusevich@gmail.com",
    description="A package for highly accurate detection of terpene synthases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pluskal-lab/EnzymeExplorer",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "get_mmseqs_based_clusters=enzymeexplorer.src.data_preparation.get_mmseqs_based_clusters:main",
            "get_uniprot_sample=enzymeexplorer.src.data_preparation.get_uniprot_sample:main",
            "get_phylogeny_based_clusters=enzymeexplorer.src.data_preparation.get_phylogeny_based_clusters:main",
            "gather_plm_embeddings=enzymeexplorer.src.embeddings_extraction.gather_required_embs:main",
            "plm_embeddings=enzymeexplorer.src.embeddings_extraction.transformer_embs:main",
            "enzyme_explorer_main=enzymeexplorer.src.enzyme_explorer_main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.10.0",
)
