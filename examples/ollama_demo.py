from depuzzle.backends.ollama import OllamaBackend


def main():
    backend = OllamaBackend(model="qwen2.5:3b")

    prompt = "Explain virtual memory in one paragraph."

    print("Starting inference...\n")

    for token in backend.generate(prompt):
        print(token, end="", flush=True)

    print("\n\nInference complete.")


if __name__ == "__main__":
    main()
