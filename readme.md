# Masterarbeit


### Fragen Tim 10 May 2024:

- (1) Wie ist die Label verteilung bei multi-label und binary?
- (2) Wie funktioniert auto model for sequence classification mit GPT/anderen generativen modellen?
- (3) Was bedeuten alle einstellungen beim Lora Adapter?
- (4) Teste es BCE loss zu gewichten:
- (5) Wo setzt man die Lora adapter am besten hin?
- (6) Mehr datensätze auch mit kürzeren context sizes:
- (7) Alle wahlen (lr, lora einstellungen etc.) mit litheratur belegen

### Antworten:

(1) Die Label Verteilung ist bei multi-label und binary wie folgt:
**TODO**

(2) Es ist auch last token pool, was bedeutet dass das embedding des letzten tokens (vor padding), 
das embedding ist nur die hidden state des letzten layers.
sources:
**TODO**

(3) Die Einstellungen beim Lora Adapter sind wie folgt:
- r:
- lora_alpha:
- lora_dropout:
**TODO**

(4) Teste den BCE loss zu gewichten liefen wie folgt:
**TODO**

(5) lora adapter bei e5-mistral-7b-instruct als vorlage für gpt neo, bei mamba nochmal in beschreibung schauen **TODO**
**TODO**

(6) Mehr datensätze auch mit kürzeren context sizes:
**TODO**

(7) Alle wahlen (lr, lora einstellungen etc.) mit litheratur belegen:
**TODO**
