""" A module with predictive models"""
from .domain_comparisons_randomforest import DomainsRandomForest
from .domain_comparisons_xgb import DomainsXgb
from .plm_randomforest import PlmRandomForest
from .plm_xgb import PlmXgb
from .plm_domain_comparison_randomforest import PlmDomainsRandomForest
from .plm_domains_mlp import PlmDomainsMLP
from .plm_domains_logistic_regression import PlmDomainsLogisticRegression

from .baselines import Blastp, Foldseek, HMM, PfamSUPFAM  # , CLEAN
