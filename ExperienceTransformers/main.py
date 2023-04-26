from transformers import pipeline


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    results = classifier(
      ["I've been waiting for a HuggingFace course my whole life.",
       "I hate this so much!",
       "fuck",
       "just so so",
       "ok",
       "yes and no",
       "yes but no"]
    )
    for result in results:
        print(result)


def zero_shot_classification():
    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(result)


def text_generation():
    generator = pipeline("text-generation")
    results = generator("In this course, we will teach you how to")
    print(results)
    results = generator(
        "In this course, we will teach you how to",
        num_return_sequences=2,
        max_length=50
    )
    print(results)


def text_generation_from_poem():
    generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
    results = generator(
        "[CLS] 万 叠 春 山 积 雨 晴 ，",
        max_length=40,
        num_return_sequences=2,
    )
    print(results)


def fill_mask():
    unmasker = pipeline("fill-mask")
    results = unmasker("This course will teach you all about <mask> models.", top_k=2)
    print(results)


def






