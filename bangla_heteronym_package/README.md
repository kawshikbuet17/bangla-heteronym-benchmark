# bangla_heteronym_package

Reads Bangla heteronyms in context: `কর` becomes `কর্‌` (tax), `করো` (do, now) or `কোরো` (do, later) depending on the
sentence. 183 heteronyms. Copy this folder into your project; it needs the packages in `requirements.txt` and a
trained checkpoint folder from the benchmark (`checkpoints/banglabert` is recommended: 99.46% test accuracy, ~12.5 ms
per sentence on a T4 GPU, ~33 ms on 4 CPU threads).

```bash
pip install -r bangla_heteronym_package/requirements.txt
cp bangla_heteronym_package/.env.example bangla_heteronym_package/.env    # optional: set the model path there
```

## Use

```python
import bangla_heteronym_package as bhp

reader = bhp.initialize("/abs/path/checkpoints/banglabert")   # once (singleton); initialize() alone uses .env / env vars
reader = bhp.get_reader()                                     # the same reader anywhere else

reader.phoneticize("আগে কর কাজটা, তারপর আবার কর।")             # 'আগে কোরো কাজটা, তারপর আবার করো।'
reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")     # 'কর্‌'
reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")  # 'করো'
reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")  # 'সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।'

reader.phoneticize_batch([text1, text2])                      # lists in, lists out, one model call per 64 words
reader.predict_batch([(sentence1, "কর"), (sentence2, "বল")])
reader.predict_marked_batch([marked1, marked2])
reader.phoneticize_marked_batch([marked1, marked2])
```

`demo.py` calls every method and shows its output: `python -m bangla_heteronym_package.demo /abs/path/checkpoints/banglabert`.

| Method | Input | Output |
|---|---|---|
| `phoneticize(text)` | any text | the text with every heteronym replaced by its reading |
| `predict(sentence, word)` | a sentence and a heteronym in it | the reading of its first occurrence |
| `predict_marked(text, marks=("[", "]"))` | text with a heteronym marked, `[কর]` | the reading of the marked word |
| `phoneticize_marked(text, marks=("[", "]"))` | text with one or more heteronyms marked | the text with only the marked words replaced and the marks removed |
| `*_batch(list)` | a list of the above | a list of the above |
| `heteronyms` | | dict: each heteronym and its readings |

## What the model is called with

In short: the model always receives **one sentence with the target word marked by `@`**, and returns **the reading
of that one word**. Whatever method you call, our code only chooses the words, builds this input, and afterwards
returns the reading or puts it back into your text.

| Model type | The model receives | The model returns | We take |
|---|---|---|---|
| Encoder (BanglaBERT etc., recommended) | the word + the marked sentence: `কর` + `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | one probability per reading: `কর্‌ 0.0003, করো 0.9699, কোরো 0.0298` | the highest: **করো** |
| Candidate scorer (`*_cand`) | the marked sentence once per reading: `... আবার @ কর @।` + `কর্‌` / `করো` / `কোরো` | one score per pair: `0.0001, 0.9995, 0.0005` | the highest: **করো** |
| Seq2seq (BanglaT5, mT5) | `phonetic: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | the whole sentence rewritten: `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার করো।` | the word at the marked position: **করো** |

In detail:

Every method ends in the same model call. The methods only differ in how they pick the words to read:

| Method | Words sent to the model |
|---|---|
| `phoneticize(text)` | every heteronym found in the text |
| `predict(sentence, word)` | the first whole-word occurrence of `word` |
| `predict_marked(text)` | the first marked word (marks removed) |
| `phoneticize_marked(text)` | every marked word (marks removed) |
| `*_batch([...])` | the same, for all inputs together |

Each chosen word becomes one item `(sentence, start, end, word)`: the sentence that contains it (text split at `।`,
`?`, `!` and newlines, at most 32 words kept on each side) and the word's character span. Items are padded into
batches of up to 64 and each batch is one model call (one forward pass, or one `generate` call for seq2seq). Before the
call the sentence is normalized (`csebuetnlp/normalizer`, BanglaBERT and BanglaT5 checkpoints only; the T5 models also
drop the ZWNJ, which their tokenizers cannot keep) and the chosen word is wrapped in the training-time marker `@ word @`.
Nothing else is added or removed.

Real calls for `predict_marked("সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার [কর]।")`, item
`("সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার কর।", 48, 50, "কর")`:

**Encoder classifier** (`checkpoints/banglabert`, tokens as the tokenizer prints them). Input is the pair (word, marked
sentence); the model returns one score per reading slot (slots the word does not have are masked); the reading is the
argmax:

```text
[CLS] কর [SEP] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP]
softmax over কর্‌ / করো / কোরো: 0.0003 / 0.9699 / 0.0298   ->  করো
```

**Candidate scorer** (`checkpoints/banglabert_cand`). One pair (marked sentence, candidate reading) per reading, all
in the same batch; one score each; the reading is the argmax:

```text
[CLS] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP] কর্ [SEP]
[CLS] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP] করো [SEP]
[CLS] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP] কোরো [SEP]
softmax: 0.0001 / 0.9995 / 0.0005   ->  করো
```

**Seq2seq** (`checkpoints/banglat5`). Input is text; the model generates the whole sentence with the word respelled
(greedy decoding); the generated word at the target's position is matched to the word's readings:

```text
input    : phonetic: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।
generated: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার করো।
->  করো
```

The other words of the sentence (here the first কর) are only context. Afterwards the reading is put back at the word's
position in your text (the `phoneticize*` methods) or returned on its own (the `predict*` methods); that part is plain
string replacement.

## Every method: model input and output

Every method with what the model received ("in") and returned ("out") for each word it read, for the three model
types (real runs of `checkpoints/banglabert`, `checkpoints/banglabert_cand` and `checkpoints/banglat5`; the same as
in `demo.py`). Your `[ ]` marks never reach the model, it sees the word marked by `@`; all rows of one call go through
the model together. `# ->` is what the method returns with BanglaBERT; the models do not always agree (see 3 and 7).
Seq2seq output has no ZWNJ (`বল্`), so the word is mapped back to the reading (`বল্‌`).

```python
reader = bhp.initialize("/abs/path/checkpoints/banglabert")

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
```

## Batch inference

Every benchmark model supports batch inference: encoder classifiers, candidate scorers and seq2seq models all run on
padded batches (64 heteronym occurrences per model call in the `*_batch` methods). Measured GPU throughput on a Tesla
T4 (full path, 200 test sentences):

| Model | Checkpoint | Type | Batch inference | Sentences/s, batch 1 | Sentences/s, batch 32 |
|---|---|---|---|---|---|
| BanglaBERT-small | `checkpoints/banglabert_small` | encoder classifier | yes | 77 | 1488 |
| BanglaBERT | `checkpoints/banglabert` | encoder classifier | yes | 79 | 1500 |
| Bangla-BERT-base | `checkpoints/bangla_bert_base` | encoder classifier | yes | 89 | 2068 |
| MuRIL-base | `checkpoints/muril_base` | encoder classifier | yes | 96 | 1976 |
| XLM-R-base | `checkpoints/xlm_roberta_base` | encoder classifier | yes | 94 | 1845 |
| IndicBERTv2-MLM | `checkpoints/indicbertv2_mlm` | encoder classifier | yes | 89 | 1972 |
| mDeBERTa-v3-base | `checkpoints/mdeberta_v3_base` | encoder classifier | yes | 38 | 235 |
| BanglaBERT-large | `checkpoints/banglabert_large` | encoder classifier | yes | 43 | 820 |
| XLM-R-large | `checkpoints/xlm_roberta_large` | encoder classifier | yes | 55 | 668 |
| BanglaT5 | `checkpoints/banglat5` | seq2seq | yes | 4 | 67 |
| mT5-small | `checkpoints/mt5_small` | seq2seq | yes | 3 | 42 |
| mT5-base | `checkpoints/mt5_base` | seq2seq | yes | 2 | 24 |
| BanglaBERT | `checkpoints/banglabert_cand` | candidate scorer | yes | 71 | 1001 |
| BanglaBERT-large | `checkpoints/banglabert_large_cand` | candidate scorer | yes | 41 | 327 |
| XLM-R-large | `checkpoints/xlm_roberta_large_cand` | candidate scorer | yes | 49 | 306 |
| BanglaT5 target-only | `checkpoints/_runs/banglat5/abl_output_target_keep/seed42` | seq2seq | yes | 12 | 205 |

If a checkpoint does not support batching (a model type not listed above, or `"batch": false` in its
`heteronym_config.json`), loading it prints a warning

```
UserWarning: <checkpoint> does not support batch inference; the *_batch methods will process their inputs one at a time
```

and the `*_batch` methods still work, one input at a time. `reader.supports_batch` tells which case applies.

## Settings

| Variable | Meaning |
|---|---|
| `BANGLA_HETERONYM_MODEL` | checkpoint folder, used when `initialize()` gets no path |
| `BANGLA_HETERONYM_DEVICE` | `cuda`, `cuda:1` or `cpu` (default: cuda if available) |

They are read from the environment, else from `.env` in this folder. A path or device given to `initialize()` wins.
`initialize()` again returns the loaded reader; to load another model, call `bhp.reset()` first.

## Input rules

- Plain Unicode Bangla. Mark a word when you want only that occurrence read: `[কর]`, no spaces inside the marks.
  `predict_marked` reads the first marked word; `phoneticize_marked` replaces every marked word.
- Do not pass marked text to `phoneticize`: it reads every heteronym and keeps the marks as punctuation.
- The marks only choose the occurrence; the model never sees them (see "What the model is called with").
- A word that is not one of the 183 heteronyms, or not in the text, raises `ValueError` (in `predict` and
  `predict_marked`); `phoneticize` leaves all other words unchanged.
- Paragraphs are fine: each heteronym is read from its own sentence (split at `।`, `?`, `!` and newlines).
- Readings with a visible hasanta keep the ZWNJ (`কর্‌` = কর + U+09CD + U+200C); keep it for the TTS front end.

## Files

`interface.py` the interface (`HeteronymReader`), `reader.py` its implementation (`HeteronymModelReader`),
`modeling.py` the model classes and checkpoint loading, `text.py` text helpers, `__init__.py` `initialize` /
`get_reader` / `reset`, `demo.py` the demo.
