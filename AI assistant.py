import json
import os
import random

NOTES_FOLDER = "course_materials"
MEMORY_FILE = "memory.json"


# ---------------------------- RAG ----------------------------
def load_notes():
    """Read every .txt file in course_materials/ and return a list of lines."""
    lines = []
    os.makedirs(NOTES_FOLDER, exist_ok=True)
    for filename in os.listdir(NOTES_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(NOTES_FOLDER, filename)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
    return lines


def find_relevant(question, lines, top_n=3):
    """Return the lines that share the most words with the question."""
    question_words = set(question.lower().split())
    scored = []
    for line in lines:
        line_words = set(line.lower().split())
        overlap = len(question_words & line_words)
        if overlap > 0:
            scored.append((overlap, line))
    scored.sort(reverse=True)
    return [line for _, line in scored[:top_n]]


# --------------------------- Memory ---------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"history": [], "scores": []}


def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def remember_chat(data, question, answer):
    data["history"].append({"q": question, "a": answer})
    save_memory(data)


def remember_score(data, topic, score, total):
    data["scores"].append({"topic": topic, "score": score, "total": total})
    save_memory(data)


# ---------------------------- Tools ----------------------------
def make_quiz(lines, num_questions=3):
    """Turn random notes into fill-in-the-blank questions."""
    lines = lines.copy()
    random.shuffle(lines)
    quiz = []
    for line in lines[:num_questions]:
        words = line.split()
        if len(words) < 3:
            continue
        answer = max(words, key=len)  # blank out the longest word
        question = line.replace(answer, "_____", 1)
        quiz.append({"question": question, "answer": answer})
    return quiz


def make_study_plan(topics, days):
    """Spread topics evenly across the number of days."""
    plan = []
    for day in range(1, days + 1):
        topic = topics[(day - 1) % len(topics)]
        plan.append(f"Day {day}: Study '{topic}' - read notes, then self-quiz.")
    return plan


# ----------------------------- Main -----------------------------
def main():
    data = load_memory()
    notes = load_notes()

    menu = """
1. Ask a question about my notes
2. Take a quiz
3. Make a study plan
4. Exit
"""

    while True:
        print(menu)
        choice = input("Choose 1-4: ")

        if choice == "1":
            question = input("Your question: ")
            relevant = find_relevant(question, notes)
            answer = "\n".join(relevant) if relevant else "I couldn't find anything about that in your notes."
            print("\nAnswer:\n" + answer)
            remember_chat(data, question, answer)

        elif choice == "2":
            quiz = make_quiz(notes)
            score = 0
            for q in quiz:
                print("\n" + q["question"])
                guess = input("Your answer: ")
                if guess.strip().lower() == q["answer"].lower():
                    print("Correct!")
                    score += 1
                else:
                    print(f"Nope, it was: {q['answer']}")
            print(f"\nScore: {score}/{len(quiz)}")
            remember_score(data, "quiz", score, len(quiz))

        elif choice == "3":
            topics = [t.strip() for t in input("Topics (comma separated): ").split(",") if t.strip()]
            days = int(input("How many days? "))
            plan = make_study_plan(topics, days)
            print("\nYour study plan:")
            for line in plan:
                print(line)

        elif choice == "4":
            print("Bye! Keep studying.")
            break

        else:
            print("Please type 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
