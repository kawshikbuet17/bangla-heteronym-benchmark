"""Calls every public method of the package once.

Run from the folder that contains bangla_heteronym_maker/:
    python -m bangla_heteronym_maker.demo /abs/path/to/checkpoints/banglabert [--device cpu]
or set BANGLA_HETERONYM_MODEL=/abs/path/to/checkpoints/banglabert and run without the path.
"""
import argparse
import time

import bangla_heteronym_maker as bhm


def show(title: str, value) -> None:
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        value = " | ".join(value)  # printed as text, so readings show their ZWNJ instead of \u200c
    print(f"\n# {title}\n{value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", help=f"checkpoint folder (default: ${bhm.MODEL_PATH_ENV})")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available)")
    args = parser.parse_args()

    # ---- initialization (singleton)
    show("is_initialized() before", bhm.is_initialized())
    start = time.perf_counter()
    reader = bhm.initialize(args.checkpoint, device=args.device)          # loads the model once
    show("initialize() took", f"{time.perf_counter() - start:.1f} s")
    show("same instance from initialize() again and get_reader()",
         bhm.initialize(args.checkpoint, device=args.device) is reader is bhm.get_reader())
    show("is_initialized() after", bhm.is_initialized())

    # ---- what the model knows
    show("number of heteronyms", len(reader.heteronyms))
    show("readings of কর", reader.heteronyms["কর"])

    # ---- 1. whole text in, corrected text out
    text = "আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব, কাল আবার দেখা হবে।"
    show("phoneticize(text)", reader.phoneticize(text))
    texts = [text, "তুমি একটু অপেক্ষা কর, আমি আসছি।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "এখানে কোনো হেটেরোনিম নেই।"]
    show("phoneticize_batch(texts)", "\n".join(reader.phoneticize_batch(texts)))

    # ---- 2. every occurrence with position and confidence
    show("readings(text)", "\n".join(map(str, reader.readings(text))))
    show("readings_batch(texts): occurrences per text", [len(r) for r in reader.readings_batch(texts)])

    # ---- 3. sentence + word in, only the word's reading out (first occurrence)
    show("predict(sentence, 'কর')", reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))
    pairs = [("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল"),
             ("তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "আঁক")]
    show("predict_batch(pairs)", reader.predict_batch(pairs))

    # ---- 4. marked text in, only the marked word's reading out
    show("predict_marked('... আবার [কর]।')", reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
    show("predict_marked with marks=('<<', '>>')", reader.predict_marked("তুমি <<বল>> নিয়ে এসো।", marks=("<<", ">>")))
    marked = ["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার কর।", "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"]
    show("predict_marked_batch(marked): first vs second কর", reader.predict_marked_batch(marked))

    # ---- errors
    for call, label in ((lambda: reader.predict_marked("কোনো চিহ্ন নেই।"), "no mark"),
                        (lambda: reader.predict("এখানে কিছু নেই।", "কর"), "word not in sentence"),
                        (lambda: reader.predict("আমি ভাত খাই।", "ভাত"), "not a known heteronym")):
        try:
            call()
        except ValueError as error:
            show(f"ValueError ({label})", error)

    # ---- batch vs a loop of single calls
    many = texts * 16
    start = time.perf_counter()
    batched = reader.phoneticize_batch(many)
    t_batch = time.perf_counter() - start
    start = time.perf_counter()
    looped = [reader.phoneticize(t) for t in many]
    t_loop = time.perf_counter() - start
    show(f"{len(many)} texts: batch vs loop", f"{t_batch * 1000:.0f} ms vs {t_loop * 1000:.0f} ms, same output: {batched == looped}")

    # ---- switching models: reset() first, then initialize() with another path
    bhm.reset()
    show("is_initialized() after reset()", bhm.is_initialized())


if __name__ == "__main__":
    main()
