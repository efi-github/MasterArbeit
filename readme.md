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

(1) Die Label Verteilung ist bei 
multi-label (allegations):
4,5437
1,1740
11,1665
3,1623
21,1558
6,1056
0,623
17,547
12,444
8,441
20,187
9,162
19,159
18,119
7,81
5,72
23,61
26,48
16,42
15,31
35,29
33,17
10,16
28,7
13,6
2,26
22,15
32,7
37,5
36,2
24,1
31,1
34,1
und binary wie folgt:
print_label_distribution_binary(violation_binary["train"].to_pandas())
[(1, 8238), (0, 762)]


(2) Es ist auch last token pool, was bedeutet dass das embedding des letzten tokens (vor padding), 
das embedding ist nur die hidden state des letzten layers.
sources:
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/gpt2/modeling_gpt2.py#L1599


(3) Die Einstellungen beim Lora Adapter sind wie folgt:
- r: dimension der matrix die trainiert wird
- lora_alpha: a value to scale the lora weights by in the addition (alpha/r) is the factor.
- lora_dropout: dropout probability of lora layers
- target_modules: the names of the modules to target, according to the paper, the more the better

(4) Teste den BCE loss zu gewichten liefen wie folgt:
Einen test durchgeführt, er lief nicht so gut wie erwartet, die gewichtung war zu groß.
Ich würde gerne zuerst oversampling probieren und dann die gewichtung nochmal abgeändert testen.

(5) lora adapter bei e5-mistral-7b-instruct als vorlage für gpt neo, bei mamba nochmal in beschreibung schauen.
see _0_mamba_vs_neo/readme.md

(6) Mehr datensätze auch mit kürzeren context sizes:
Eine vorherige version der daten, mit max länge 2.6k tokens: https://archive.org/details/ECHR-ACL2019
(noch nicht getestet)
Ideen code zu microcode, microcode zu code, könnte auch interessant sein.
Schwierigkeiten:
- mamba kann sich nur so viel merken, compiler anschauen wie das da gemacht wird, wie viel speicher sie brauchen.
- kann man vielleicht ein look back implementieren?

Ideen author identifikation:
https://downloads.webis.de/pan/publications/papers/kestemont_2018.pdf
- dafür sollte es ausreichen, vielleicht eine abwandlung des siamese networks, mit einem kleinen model zur identifikation.
- vielleicht datetime informationen benutzen und "zeit" profile erstellen (also praktisch s)

(7) Alle wahlen (lr, lora einstellungen etc.) mit litheratur belegen:
see _0_mamba_vs_neo/readme.md
nochmal mit der erhöhten r zahl ausprobieren, wenn das aus der litheratur besser passt, das nehemen.



### Fragen Tim 3 June 2024:

- (1) Mehr datensätze (wir haben uns in zoom https://arxiv.org/pdf/2109.04712v2 und https://paperswithcode.com/sota/multi-label-text-classification-on-reuters-1)
https://paperswithcode.com/sota/multi-label-text-classification-on-eur-lex
- (2) Bert nochmal base hinzufügen und vergleichen
- (3) das als motivation für die anderen modelle (problem statement)
- (4) entscheiden für eine token länge (4k/8k) 24k is too much
EXPOSE ^
- (5) ansatz E5 für few shot classification ausarbeiten und nächstes mal besprechen