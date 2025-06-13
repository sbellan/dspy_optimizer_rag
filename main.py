import dspy
import json

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

def main():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    # Load trainset
    trainset = []
    with open("trainset.jsonl", "r") as f:
        for line in f:
            trainset.append(dspy.Example(**json.loads(line)).with_inputs("question"))

    # Load valset
    valset = []
    with open("valset.jsonl", "r") as f:
        for line in f:
            valset.append(dspy.Example(**json.loads(line)).with_inputs("question"))
            
    tp = dspy.MIPROv2(
       metric=dspy.evaluate.answer_exact_match,
        auto="light",
        num_threads=16
    )
    
    # dspy.cache.load_memory_cache("./memory_cache.pkl")
    
    optimized_react = tp.compile(
        react,
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
    )
    
    # Final instruction and few-shots
    print(optimized_react.react.signature)
    print(optimized_react.react.demos)
    
    # Evaluate
    evaluator = dspy.Evaluate(
        metric=dspy.evaluate.answer_exact_match,
        devset=valset,
        display_table=True,
        display_progress=True,
        num_threads=24,
    )
    original_score = evaluator(react)
    print(f"Original score: {original_score}")
    
    optimized_score = evaluator(optimized_react)
    print(f"Optimized score: {optimized_score}")


if __name__ == "__main__":
    main()
