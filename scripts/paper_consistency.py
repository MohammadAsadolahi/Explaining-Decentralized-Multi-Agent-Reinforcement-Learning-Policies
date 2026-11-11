"""Audit the paper's published numbers without pretending they are measured."""
import json
from dmarl_explain.paper_consistency import check_published_tables

if __name__ == '__main__':
    print(json.dumps(check_published_tables(), indent=2, sort_keys=True))
