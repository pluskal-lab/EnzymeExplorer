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
            "prepare_dataset=enzymeexplorer.src.data_preparation.prepare_dataset:main",
            "detect_domains=enzymeexplorer.src.structure_processing.domain_detections:main",
            "run_hac_domain_clustering=scripts.run_hac_domain_clustering:main",
            "run_dynamic_tree_cut_sweep=scripts.run_dynamic_tree_cut_sweep:main",
            "run_domain_subtype_labeling=scripts.run_domain_subtype_labeling:main",
            "structural_features=enzymeexplorer.src.structure_processing.get_structural_features:main",
            "predict_with_structures=enzymeexplorer.src.prediction.predict_with_structures:main",
            "predict_sequences_only=enzymeexplorer.src.prediction.predict_sequences_only:main",
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
    install_requires=[
        "configargparse",
    ],
    python_requires="==3.10.0",
)
