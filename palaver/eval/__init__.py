"""Eval harness scoring extraction quality across model legs (Task 3.5).

Scores what `palaver.extract`'s model-assisted extraction layer produces
against labelled fixtures -- never what `palaver.extract.persist` does with
it afterwards. See `palaver.eval.harness` for the orchestration and scoring
logic and `palaver.cli.eval` for the `palaver eval` subcommand.
"""
