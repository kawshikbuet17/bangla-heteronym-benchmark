"""Every method of the package, with its output (from checkpoints/banglabert).

Run from the folder that contains bangla_heteronym_package/:
    python -m bangla_heteronym_package.demo /abs/path/to/checkpoints/banglabert
or without the path when BANGLA_HETERONYM_MODEL is set (environment or bangla_heteronym_package/.env).
"""
import sys

import bangla_heteronym_package as bhp

# 1. Load the model once. Later initialize() / get_reader() calls return the same reader (singleton).
reader = bhp.initialize(sys.argv[1] if len(sys.argv) > 1 else None)   # device="cpu" / "cuda" / "cuda:1" optional
print(bhp.get_reader() is reader)
# -> True

# 2. The known heteronyms and their readings
print(len(reader.heteronyms), *reader.heteronyms["কর"])
# -> 183 কর্‌ করো কোরো

# 3. Text in, corrected text out
print(reader.phoneticize("আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।"))
# -> আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।

# 4. Many texts in one call
for text in reader.phoneticize_batch(["তুমি একটু অপেক্ষা কর, আমি আসছি।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "এখানে কোনো হেটেরোনিম নেই।"]):
    print(text)
# -> তুমি একটু অপেক্ষা করো, আমি আসছি।
# -> তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
# -> এখানে কোনো হেটেরোনিম নেই।

# 5. Sentence + word in, only the word's reading out (first occurrence of the word)
print(reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))
# -> কর্‌

# 6. Many (sentence, word) pairs in one call
print(*reader.predict_batch([("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল")]))
# -> কর্‌ বল্‌

# 7. Marked word in, only its reading out (use it when the word occurs more than once)
print(reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
# -> করো

# 8. Other marks, if [ ] can occur in your text
print(reader.predict_marked("মাঠে যাওয়ার সময় <<বল>> নিয়ে যেও।", marks=("<<", ">>")))
# -> বল্‌

# 9. Many marked texts in one call (here the first and the second কর of the same sentence)
print(*reader.predict_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার কর।", "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"]))
# -> কর্‌ করো

# 10. Marked text in, whole text out: only the marked words are replaced, the marks are removed
print(reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
# -> সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।

# 11. Many marked texts in one call; a text can have several marked words
for text in reader.phoneticize_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার [কর]।", "মাঠে যাওয়ার সময় [বল] নিয়ে যেও।"]):
    print(text)
# -> সরকার কর্‌ বাড়িয়েছে, তাই হিসাবটা আবার করো।
# -> মাঠে যাওয়ার সময় বল্‌ নিয়ে যেও।

# 12. Wrong input raises ValueError
try:
    reader.predict("আমি ভাত খাই।", "ভাত")
except ValueError as error:
    print(error)
# -> 'ভাত' is not one of the 183 known heteronyms

# 13. To load another model: reset(), then initialize() with the other path
bhp.reset()
