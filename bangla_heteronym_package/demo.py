"""Every method of the package, with its output (from checkpoints/banglabert).

Under each call, "model in" / "model out" is what the model itself received and returned: one row per word to read,
the word + its sentence with the word marked by @ (your [ ] marks are removed), and one probability per reading.
All rows of one call go through the model together (one batch); "# ->" is what the method returns to you.
(BanglaBERT is an encoder. A candidate checkpoint gets one row per reading, "<marked sentence> + <reading>", and returns
one score per row; a seq2seq checkpoint gets "phonetic: <marked sentence>" and returns the sentence with the word
respelled, from which the word at the marked position is taken.)

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
#   model in:  কর + আগে @ কর @ কাজটা, তারপর আবার কর।
#   model out: কর্‌ 0.2683, করো 0.0227, কোরো 0.7090 -> কোরো
#   model in:  কর + আগে কর কাজটা, তারপর আবার @ কর @।
#   model out: কর্‌ 0.0013, করো 0.7762, কোরো 0.2225 -> করো
#   model in:  বল + বিকেলে মাঠে @ বল @ নিয়ে যাব।
#   model out: বল্‌ 0.9998, বলো 0.0002 -> বল্‌
# -> আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।

# 4. Many texts in one call
for text in reader.phoneticize_batch(["তুমি একটু অপেক্ষা কর, আমি আসছি।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "এখানে কোনো হেটেরোনিম নেই।"]):
    print(text)
#   model in:  কর + তুমি একটু অপেক্ষা @ কর @, আমি আসছি।
#   model out: কর্‌ 0.0003, করো 0.9464, কোরো 0.0533 -> করো
#   model in:  আঁক + তুমি গ্রাফ পেপারে সরলরেখাটা @ আঁক @।
#   model out: আঁক্‌ 0.0003, আঁকো 0.9997 -> আঁকো
#   (the third text has no heteronym, so it is not sent to the model)
# -> তুমি একটু অপেক্ষা করো, আমি আসছি।
# -> তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
# -> এখানে কোনো হেটেরোনিম নেই।

# 5. Sentence + word in, only the word's reading out (first occurrence of the word)
print(reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))
#   model in:  কর + সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#   model out: কর্‌ 0.9868, করো 0.0002, কোরো 0.0130 -> কর্‌
# -> কর্‌

# 6. Many (sentence, word) pairs in one call
print(*reader.predict_batch([("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল")]))
#   model in:  কর + সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#   model out: কর্‌ 0.9868, করো 0.0002, কোরো 0.0130 -> কর্‌
#   model in:  বল + বিকেলে মাঠে @ বল @ নিয়ে যাব।
#   model out: বল্‌ 0.9998, বলো 0.0002 -> বল্‌
# -> কর্‌ বল্‌

# 7. Marked word in, only its reading out (use it when the word occurs more than once)
print(reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   model in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#   model out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
# -> করো

# 8. Other marks, if [ ] can occur in your text
print(reader.predict_marked("মাঠে যাওয়ার সময় <<বল>> নিয়ে যেও।", marks=("<<", ">>")))
#   model in:  বল + মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#   model out: বল্‌ 0.9997, বলো 0.0003 -> বল্‌
# -> বল্‌

# 9. Many marked texts in one call (here the first and the second কর of the same sentence)
print(*reader.predict_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার কর।", "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"]))
#   model in:  কর + সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#   model out: কর্‌ 0.9857, করো 0.0002, কোরো 0.0141 -> কর্‌
#   model in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#   model out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
# -> কর্‌ করো

# 10. Marked text in, whole text out: only the marked words are replaced, the marks are removed
print(reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   model in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#   model out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
# -> সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।

# 11. Many marked texts in one call; a text can have several marked words
for text in reader.phoneticize_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার [কর]।", "মাঠে যাওয়ার সময় [বল] নিয়ে যেও।"]):
    print(text)
#   model in:  কর + সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#   model out: কর্‌ 0.9857, করো 0.0002, কোরো 0.0141 -> কর্‌
#   model in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#   model out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   model in:  বল + মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#   model out: বল্‌ 0.9997, বলো 0.0003 -> বল্‌
# -> সরকার কর্‌ বাড়িয়েছে, তাই হিসাবটা আবার করো।
# -> মাঠে যাওয়ার সময় বল্‌ নিয়ে যেও।

# 12. Wrong input raises ValueError
try:
    reader.predict("আমি ভাত খাই।", "ভাত")
except ValueError as error:
    print(error)
#   (checked before the model: the model is not called)
# -> 'ভাত' is not one of the 183 known heteronyms

# 13. To load another model: reset(), then initialize() with the other path
bhp.reset()
