"""Every method of the package, with its output (from checkpoints/banglabert).

Under each call: what the model itself received ("in") and returned ("out") for each word it read, for the three model
types, each from its real checkpoint. Your [ ] marks never reach the model; it sees the word marked by @. All rows of
one call go through the model together (one batch).
  encoder (banglabert):         in = word + marked sentence;  out = one probability per reading -> the highest
  candidate (banglabert_cand):  in = marked sentence + reading, one row per reading;  out = one probability per row
  seq2seq (banglat5):           in = "phonetic: " + marked sentence;  out = the sentence rewritten -> the word at the
                                marked position (T5 drops the ZWNJ, e.g. বল্, so it is mapped back to the reading বল্‌)
"# ->" is what the method returns with checkpoints/banglabert. The models do not always agree (see 3 and 7).

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
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + আগে @ কর @ কাজটা, তারপর আবার কর।
#     out: কর্‌ 0.2683, করো 0.0227, কোরো 0.7090 -> কোরো
#     in:  কর + আগে কর কাজটা, তারপর আবার @ কর @।
#     out: কর্‌ 0.0013, করো 0.7762, কোরো 0.2225 -> করো
#     in:  বল + বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বল্‌ 0.9998, বলো 0.0002 -> বল্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  আগে @ কর @ কাজটা, তারপর আবার কর। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.0555, করো 0.9440, কোরো 0.0004 -> করো
#     in:  আগে কর কাজটা, তারপর আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.0084, করো 0.9821, কোরো 0.0095 -> করো
#     in:  বিকেলে মাঠে @ বল @ নিয়ে যাব। + each of বল্‌ / বলো
#     out: বল্‌ 1.0000, বলো 0.0000 -> বল্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: আগে @ কর @ কাজটা, তারপর আবার কর।
#     out: আগে করো কাজটা, তারপর আবার করো। -> করো
#     in:  phonetic: আগে কর কাজটা, তারপর আবার @ কর @।
#     out: আগে করো কাজটা, তারপর আবার করো। -> করো
#     in:  phonetic: বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বিকেলে মাঠে বল্ নিয়ে যাব। -> বল্‌
# -> আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।

# 4. Many texts in one call
for text in reader.phoneticize_batch(["তুমি একটু অপেক্ষা কর, আমি আসছি।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "এখানে কোনো হেটেরোনিম নেই।"]):
    print(text)
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + তুমি একটু অপেক্ষা @ কর @, আমি আসছি।
#     out: কর্‌ 0.0003, করো 0.9464, কোরো 0.0533 -> করো
#     in:  আঁক + তুমি গ্রাফ পেপারে সরলরেখাটা @ আঁক @।
#     out: আঁক্‌ 0.0003, আঁকো 0.9997 -> আঁকো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  তুমি একটু অপেক্ষা @ কর @, আমি আসছি। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.0001, করো 0.9966, কোরো 0.0033 -> করো
#     in:  তুমি গ্রাফ পেপারে সরলরেখাটা @ আঁক @। + each of আঁক্‌ / আঁকো
#     out: আঁক্‌ 0.0000, আঁকো 1.0000 -> আঁকো
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: তুমি একটু অপেক্ষা @ কর @, আমি আসছি।
#     out: তুমি একটু অপেক্ষা করো, আমি আসছি। -> করো
#     in:  phonetic: তুমি গ্রাফ পেপারে সরলরেখাটা @ আঁক @।
#     out: তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো। -> আঁকো
#   (the third text has no heteronym, so it is not sent to the model)
# -> তুমি একটু অপেক্ষা করো, আমি আসছি।
# -> তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
# -> এখানে কোনো হেটেরোনিম নেই।

# 5. Sentence + word in, only the word's reading out (first occurrence of the word)
print(reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: কর্‌ 0.9868, করো 0.0002, কোরো 0.0130 -> কর্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 1.0000, করো 0.0000, কোরো 0.0000 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: সময়মতো কর্ না দিলে জরিমানা গুনতে হবে। -> কর্‌
# -> কর্‌

# 6. Many (sentence, word) pairs in one call
print(*reader.predict_batch([("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল")]))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: কর্‌ 0.9868, করো 0.0002, কোরো 0.0130 -> কর্‌
#     in:  বল + বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বল্‌ 0.9998, বলো 0.0002 -> বল্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 1.0000, করো 0.0000, কোরো 0.0000 -> কর্‌
#     in:  বিকেলে মাঠে @ বল @ নিয়ে যাব। + each of বল্‌ / বলো
#     out: বল্‌ 1.0000, বলো 0.0000 -> বল্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: সময়মতো কর্ না দিলে জরিমানা গুনতে হবে। -> কর্‌
#     in:  phonetic: বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বিকেলে মাঠে বল্ নিয়ে যাব। -> বল্‌
# -> কর্‌ বল্‌

# 7. Marked word in, only its reading out (use it when the word occurs more than once)
print(reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
# -> করো

# 8. Other marks, if [ ] can occur in your text
print(reader.predict_marked("মাঠে যাওয়ার সময় <<বল>> নিয়ে যেও।", marks=("<<", ">>")))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  বল + মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#     out: বল্‌ 0.9997, বলো 0.0003 -> বল্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও। + each of বল্‌ / বলো
#     out: বল্‌ 1.0000, বলো 0.0000 -> বল্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#     out: মাঠে যাওয়ার সময় বল্ নিয়ে যেও। -> বল্‌
# -> বল্‌

# 9. Many marked texts in one call (here the first and the second কর of the same sentence)
print(*reader.predict_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার কর।", "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"]))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#     out: কর্‌ 0.9857, করো 0.0002, কোরো 0.0141 -> কর্‌
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 1.0000, করো 0.0000, কোরো 0.0000 -> কর্‌
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> কর্‌
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
# -> কর্‌ করো

# 10. Marked text in, whole text out: only the marked words are replaced, the marks are removed
print(reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
# -> সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।

# 11. Many marked texts in one call; a text can have several marked words
for text in reader.phoneticize_marked_batch(["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার [কর]।", "মাঠে যাওয়ার সময় [বল] নিয়ে যেও।"]):
    print(text)
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#     out: কর্‌ 0.9857, করো 0.0002, কোরো 0.0141 -> কর্‌
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#     in:  বল + মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#     out: বল্‌ 0.9997, বলো 0.0003 -> বল্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 1.0000, করো 0.0000, কোরো 0.0000 -> কর্‌
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#     in:  মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও। + each of বল্‌ / বলো
#     out: বল্‌ 1.0000, বলো 0.0000 -> বল্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার @ কর @ বাড়িয়েছে, তাই হিসাবটা আবার কর।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> কর্‌
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
#     in:  phonetic: মাঠে যাওয়ার সময় @ বল @ নিয়ে যেও।
#     out: মাঠে যাওয়ার সময় বল্ নিয়ে যেও। -> বল্‌
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
