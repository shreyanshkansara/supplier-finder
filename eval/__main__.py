"""python -m eval  ->  parser/completeness eval, then retrieval/ranking eval."""

from eval import run_eval, run_parse_eval

if __name__ == "__main__":
    run_parse_eval.main()
    run_eval.main()
