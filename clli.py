#!/usr/bin/env python3
"""
cli.py — command-line interface for the AI Study Assistant.

Usage:
    python cli.py ingest <path>                 # ingest course materials (.txt/.md/.pdf)
    python cli.py chat                          # interactive chat with the agent
    python cli.py ask "<question>"               # one-off question
    python cli.py quiz "<topic>" [--num N] [--difficulty easy|medium|hard]
    python cli.py plan "<topic>" --days N [--subtopics "a,b,c"]
"""
from __future__ import annotations

import argparse
import sys

from study_assistant import StudyAssistantAgent, VectorStore, load_and_chunk


def cmd_ingest(args: argparse.Namespace) -> None:
    print(f"Scanning {args.path} ...")
    chunks = load_and_chunk(args.path)
    if not chunks:
        print("No supported files (.txt, .md, .pdf) found.")
        return
    store = VectorStore()
    n = store.add_chunks(chunks)
    print(f"Ingested {n} chunks into the vector store.")
    print(f"Sources now indexed: {', '.join(store.list_sources())}")


def cmd_chat(_: argparse.Namespace) -> None:
    agent = StudyAssistantAgent()
    print("AI Study Assistant — type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue
        print(f"\nassistant> {agent.chat(user_input)}\n")


def cmd_ask(args: argparse.Namespace) -> None:
    print(StudyAssistantAgent().chat(args.question))


def cmd_quiz(args: argparse.Namespace) -> None:
    prompt = f"Quiz me on {args.topic}, {args.num} questions, {args.difficulty} difficulty."
    print(StudyAssistantAgent().chat(prompt))


def cmd_plan(args: argparse.Namespace) -> None:
    subtopics = f" Subtopics: {args.subtopics}." if args.subtopics else ""
    prompt = f"Create a {args.days}-day study plan for {args.topic}.{subtopics}"
    print(StudyAssistantAgent().chat(prompt))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Study Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest course materials into the vector store")
    p.add_argument("path")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chat", help="Interactive chat session")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", help="Ask a single question")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("quiz", help="Generate a quiz on a topic")
    p.add_argument("topic")
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.set_defaults(func=cmd_quiz)

    p = sub.add_parser("plan", help="Generate a study plan")
    p.add_argument("topic")
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--subtopics", default=None)
    p.set_defaults(func=cmd_plan)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
